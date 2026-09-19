# Measurement history

> **Historical record — not instructions.** It preserves every measurement record of the former
> `CLAUDE.md` §7, the phase records of the former `CLAUDE.md` §8, and measurement facts that
> existed only in the former handoff, moved **verbatim** from base
> `ae42cb01677f94868b2873008d87be677e31f0c8`. Verdicts are quoted as recorded when reviewed;
> this document re-reviews nothing. Statements inside a record about what was "next",
> "pending" or "authorized" were true on their own date only. Inside moved text a bare `§N`
> means that former document's section
> ([compatibility index](../../CLAUDE.md#8-compatibility-index-for-older-references)); former
> handoff `§3d`–`§3q` sections are the sub-sections of this document and of
> [`decisions.md`](decisions.md). How to plan and review runs:
> [`experiments.md`](../workflows/experiments.md).

## 1. Run registry

| Run | Measured code SHA | Location as recorded | Verdict as recorded |
|---|---|---|---|
| first real post-B3 probe (easy pre-FD cell) | `a3f0838616990987bcb8a51665fa75d84edf5952` | not recorded | reviewed short probe; **not a baseline** and not evidence about the fuel-damage cell |
| first final-cell short probe | `238062d7d284334432d9c39d7543fb0bbf39ea7c` | `training_output_20260815_173029` | operability only; scientifically inconclusive (exposed Defects A–C) |
| corrected-cell short probe | `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` | `training_output_20260816_162130` | originally `VALID`, **superseded** by `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR` |
| first long baseline | `c30b6982ba605d60976cc303256da4b5528b0e63` | `training_output_long_baseline_100x8_seed0` | engineering `REQUEST FIXES`; scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED` |
| Phase-A long baseline (rerun) | `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` | `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` | `APPROVE — VALID MEASUREMENT` |
| FD-VARIABLE-SEVERITY-v1 precursor | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640` | `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT` |
| FD-VARIABLE-SEVERITY-v1 actor-only baseline | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | `C:\Users\Itama\f7r2` | `APPROVE — VALID MEASUREMENT`; primary finding negative |
| old fixed-cell CTDE measurement | not recorded | not recorded | executed; **out of scope**; nothing recorded |
| Task 5A / Task 5B | Task 5B: `4af6c5aa5dd28072692bfda63282964b55010aae` | not recorded | `APPROVE — VALID ENGINEERING VALIDATION` — **not measurements** |
| GENERALIZED-V1 R1 (actor-only) | `4af6c5aa5dd28072692bfda63282964b55010aae` | not recorded | `APPROVE — VALID MEASUREMENT`; primary FD finding negative |
| R1 diagnostic replay | R1's SHA | bundle SHA-256 `812ff43322e134e9a7ca31720007393ff1220ba50c35955b2a724b30d4d5d792` | engineering / analysis evidence — **not a measurement** |
| aborted P1 arm | not recorded | not recorded | `ABORTED / DO NOT RESUME` — not a measurement |
| fresh deterministic-P1 arm | `ae1941035991df4719df212c4b5dd07db89aee4a` | not recorded | accepted as a valid measurement; negative primary MILD-vs-SEVERE result |
| GENERALIZED-V2 benchmark preflight (produced the external manifest) | `ae42cb01677f94868b2873008d87be677e31f0c8` — **producer-recorded** in `benchmark_preflight_report.json:/provenance/git` with `dirty = false`; not an external attestation (§8.11) | external local `C:\Users\Itama\PycharmProjects\graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0`; reviewed via temporary review PR #67, head `7f56338cde6aacfa59a52399b2378b98a62ea3aa` (not for merge) | provenance package `APPROVE — provenance package correctness / evidence review` (not a scientific-validity verdict); historical research authorization and historical prior review `NOT PRESERVED / NOT PROVEN` |
| GENERALIZED-V2 development R1 — actor-only | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #61, head `1375a881637a9a32721a1630f598adc571422a47`; local original `C:\Users\Itama\PycharmProjects\graph_rl_v2_actor_only_dev_r1_seed3000000_ae42cb0` | no verdict recorded in accessible artifacts; an input to the 2026-09-15 development interpretation (§8) |
| GENERALIZED-V2 development R1 — CTDE | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #62, head `b2bbe7a6235c3b9255106826cfb268af7e73f72d`; local original `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_r1_seed3000000_ae42cb0` | PR #62 records a prior GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT`; not independently verified |
| GENERALIZED-V2 CTDE diagnostic — `smallbatch` | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #64, head `90516d51beeddacded2b89a321d14291e411f2b0`; local original `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_smallbatch_seed3000000_ae42cb0` | **development diagnostic run, not a confirmatory measurement**; accounting `PASS` (§8) |
| GENERALIZED-V2 CTDE diagnostic — `largebatch` | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #64 (same head); local original `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_largebatch_seed3000000_ae42cb0` | **development diagnostic run, not a confirmatory measurement**; accounting `PASS` (§8) |
| GENERALIZED-V2 CTDE diagnostic — `fd80` | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #64 (same head); local original `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_fd80_seed3000000_ae42cb0` | **development diagnostic run, not a confirmatory measurement**; accounting `PASS` (§8) |
| GENERALIZED-V2 matched immediate-FD wake extraction (all five runs above) | reads the five runs' recorded artifacts; executes no project code | temporary review PR #65, head `d565174e4ecc25eb60a4dd021e1a20025f55f07f` | **read-only analysis package, not a measurement and not a run**; extraction integrity `APPROVE` (§8) |
| GENERALIZED-V2 semantic-action actor-only development R1 (`semantic_k_plus_2_logmeanexp_v1`) | `d4e9f3721e6d151c00be3fe93c3d149df9d31965` | evidence PR #71, exact candidate `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670` (not for merge); local original `C:\Users\Itama\PycharmProjects\graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37` | GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT` (2026-09-16, as transferred in the user-approved documentation packet); **development only**; final primary endpoint effectively zero, transient separation at updates 75–150 (§10) |
| GENERALIZED-V2 semantic-action CTDE actor-gradient diagnostic A — `p = 0.5` (150 updates) | `6ed964a1abd09de2130aee3d0d314c8f32165056` — PR #74 approved head, **unmerged when measured** | local original `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_r1_seed3000000_6ed964a`; compact index `research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/` (PR #74) | GPT verdict `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` (2026-09-17, as transferred in the user-approved documentation packet); **development diagnostic only**; no meaningful severity-conditioned behaviour (§11) |
| GENERALIZED-V2 semantic-action CTDE actor-gradient diagnostic B — FD100 intervention (150 updates) | `6ed964a1abd09de2130aee3d0d314c8f32165056` — PR #74 approved head, **unmerged when measured** | local original `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a`; compact index as above | GPT verdict `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` (2026-09-17, as transferred in the user-approved documentation packet); **development diagnostic only**; probability-level separation acquired at update 75, not retained (§11) |
| GENERALIZED-V2 role-only acting-ego CTDE development diagnostic — FD100 configuration (100 updates) | `68055e39768d5fa601e5960a9f08823b9e65c08f` — PR #75 approved implementation head, **unmerged when measured** | local original `C:\gruns\graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3`; compact package `research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/` on PR #75 | GPT verdict `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` (2026-09-19, as transferred in the user-approved packet); **development diagnostic only, cross-version** against diagnostic B; critic localization improved, held-out acquisition / retention not improved (§12) |

Locations above are **as recorded on their own dates** and are not rewritten. Since 2026-09-15 the
local artifacts live in the local archive under `C:\gra\`, and PRs #61, #62, #64, #65 and #67 are
closed with their branches deleted; current locations and archive identities are in §9.

## 2. Measurement records

- `a3f0838` — **First real post-B3 instrumented probe — CLOSED / REVIEWED MEASUREMENT.**
  The Grade-A measurement is attributable to exact clean code SHA
  `a3f0838616990987bcb8a51665fa75d84edf5952` on the measurement-only branch
  `task/b4-first-instrumented-probe`; no tracked file changed and no PR or candidate
  commit existed. The exact cell was two iterations × four scheduled train episodes,
  seeds `[0,8)`, plus the same fixed held-out seeds `[1000000,1000004)` before the first
  update and after two completed updates. Provenance was complete
  (`git.available=true`, exact SHA, `dirty=false`, Windows / `nlp_env`, vendored BLADE,
  BONMIN available), the process completed normally in 79.21 s, all six expected
  artifacts existed, and `run_summary.json:accounting_reconciled=true`. Measured:
  `pre_update=-0.4999997395829586` over **4/4**; training **7/8** successful with every
  successful episode producing wakes, **24 transitions**, two productive iterations and
  two PPO updates; one attempt failed exactly once — train seed 2 at `setup`, because B2
  produced two placements for three requested hidden targets after the static solve left
  one ego without a non-empty route; final `post_update=5.000007394910353e-7` over
  **4/4** (numerical zero). This proves learning headroom, usable data yield and a working
  update/eval loop. It is a SHORT PROBE, not a baseline. The original
  `kills_mean` / `eval_kills_mean` fields counted `(ego_id,target_id)` confirmations and
  are invalid as unique-target counts; reward, wake, failure, transition and PPO evidence
  remain valid because reward already deduplicated by target id and the count was not a
  PPO input. Evidence SHA-256:
  `run_config.json=36ec89cdb93f89c0b6e40163491159bf2045235b86b2fad47fe03f2f86141237`,
  `train_records.jsonl=af4ec1851425fbcd0330651c05e384d0e44dad67f8aa1f56080543d8247ad82d`,
  `eval_records.jsonl=2c972efaf85d465ab4f2ffce164ba19ac2a6c189db1e2faf83de6b0d201a7439`,
  `episode_failures.jsonl=32d51d2d2ec017491f2fbe6bf133e103361752ced66ba39aac51e9b35b03a08e`,
  `run_summary.json=d2e24714eecdf48bd5f1478ba1c119f405bef5d82067840776daa26dd4270c80`,
  `training_plot.png=c6dec3ac99c5bd35fe627f77b2e97f432cb33235ce07f7efed8f0c05d7a9521b`.
- **CORRECTED-CELL BOUNDED SHORT PROBE — EXECUTED / INDEPENDENTLY REVIEWED / VERDICT
  SUPERSEDED: SCIENTIFICALLY INCONCLUSIVE.** **READ THE SUPERSEDING VERDICT BELOW BEFORE
  ANY NUMBER IN THIS ENTRY.** Everything recorded here about run IDENTITY, invocation,
  provenance, accounting, artifact completeness and playback is PRESERVED HISTORICAL
  EVIDENCE and unchanged; what changed is the SCIENTIFIC verdict, which the later
  roster-integrity review invalidated (the superseding-verdict paragraph at the end of this
  entry, and the roster fix's own lock further below). The measurement is attributable to
  exact clean code SHA
  `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` (committed `2026-08-16T15:26:55+03:00`), the
  `main` head produced by the Defect-C documentation merge (PR #22). Per §7's hash
  convention this entry is keyed to the MEASURED CODE SHA; the documentation commit that
  creates it, and the merge that integrates it, cannot name their own SHAs and are
  deliberately not invented here. **No tracked file changed for the measurement** — it is
  a run of merged code, not a candidate.
  **RUN IDENTITY.** Run directory `training_output_20260816_162130`. Exactly ONE
  invocation, native exit code **0**:
  `conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json`.
  The tracked checkout was clean before and after. Preset blob
  `3c85e5bdc780600fe1ee528b3e35fc71591fe4b7`,
  `config_source.resolved_from = config_file`, `cli_overrides = []` — so the probe ran the
  reviewed preset with NO typed override and NO ad-hoc knob. Provenance complete:
  `git.available=true`, exact SHA, `branch=main`, `dirty=false`, `dirty_path_count=0`,
  Windows / `nlp_env` (CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`,
  `difficulty.factor = fuel_damage_baseline_v1` with `aircraft_penalty_coeff = 2.25` and
  `reward.formula_changed = false`. **Elapsed time is TWO DISTINCT QUANTITIES and they are
  never merged:** the harness's own `run_summary.json:run_seconds = 204.50847799982876`,
  and the externally measured invocation wall clock of **223.117 s** (the preserved probe
  runner's `timing.txt`: `PROBE_START_UTC = 2026-08-16T13:21:24.1839202Z` →
  `PROBE_END_UTC = 2026-08-16T13:25:07.3008671Z`). The harness figure excludes process
  start-up, `conda run` dispatch, imports and interpreter teardown, so it is NOT the wall
  clock and must not be labelled as one. **No code, configuration, preset or
  research-semantic change accompanied the run.**
  **VALIDITY VERDICT AS ORIGINALLY REVIEWED — `VALID MEASUREMENT / CORRECTED SHORT-PROBE
  PASS`. THAT VERDICT IS SUPERSEDED AND NO LONGER HOLDS** (see the superseding-verdict
  paragraph at the end of this entry); it is quoted here only so the review history stays
  legible. As originally judged, against the pre-declared validity gate and NOT against
  whether reward improved: exact clean Git
  provenance is complete; `run_summary.json:accounting_reconciled = true`; no
  INFRASTRUCTURE failure occurred (no `_VisualArtifactError`, and both recorded failures
  sit inside the `generation`/`setup`/`run`/`reward` episode taxonomy); BOTH evaluation
  rounds carry **4/4 complete matched pairs**; and PPO-update and artifact evidence are
  complete. At the time of that review this was read as closing the research-validity
  gate the FIRST probe exposed; **that reading is SUPERSEDED — the gate was not closed by
  this run** (end of this entry). What DOES survive unaffected: Defects A, B and C remain
  CLOSED / APPROVED / MERGED and are OPERATIONALLY WITNESSED in real playback from this run
  rather than in proof tests alone — the roster defect concerns which targets the
  MEASUREMENT counted, not whether those three corrections work.
  **ACCOUNTING — every denominator explicit, `skip_and_account_v1` unchanged.**
  **24 scheduled attempts** (8 train + 16 eval), **22 successful**, **2 failed, both at
  `setup`**. By condition: **clean 11 attempted / 10 successful / 1 failed**; **damaged 13
  attempted / 12 successful / 1 failed**. Training 8 attempted / 6 successful / 2 failed,
  all 6 successes producing wakes; evaluation **16/16 successful, 0 failed**.
  `accounting_reconciled = true`. **No retry, no substitution, no seed-band shift** — each
  failed seed was attempted once and recorded once:
  - train seed 2, **damaged**, `setup`, `RuntimeError` — B2 produced only two placements
    for `n_hidden=3`, because the static solve left one of the three egos without a
    non-empty route (the known exact-cardinality behaviour, §8);
  - train seed 4, **clean**, `setup`, `EpisodeRosterError` — one t=0 known target was
    absent from the executed world, so the roster would not have covered what runs.

  **FD FIRING RATE — state it against the right denominator.** The fuel-damage event fired
  and woke its selected ego in **12/12 successfully completed damaged episodes** (4 train +
  8 eval), which is **12/13 SCHEDULED damaged attempts**: the damaged seed-2 attempt failed
  during `setup`, before an event could exist. It must never be described as 12/12
  scheduled damaged attempts.
  **MATCHED HELD-OUT EVALUATION — the only within-seed claim is the paired delta.** Both
  rounds ran the same fixed band `[1000000,1000004)`, each seed twice (`forced_clean` +
  `forced_damaged`), 8 attempted / 8 successful per round.
  - `pre_update` (`updates_completed = 0`): **4/4 complete pairs**; clean mean
    `-0.4999997395829586` over 4; damaged mean `-0.8749999192707323` over 4; paired delta
    `-0.37500017968777366` over 4/4 pairs; **eval deaths 4**; unique targets confirmed
    mean `3.00`; meta-action mix `PLAN_COMPLIANCE 52 / OPPORTUNISTIC_ENGAGEMENT 0 /
    SELF_PRESERVATION_ABORT 0`.
  - `post_update` (`updates_completed = 2`): **4/4 complete pairs**; clean mean
    `-0.12499955989518505` over 4; damaged mean `-0.4583330529509838` over 4; paired delta
    `-0.33333349305579874` over 4/4 pairs; **eval deaths 0**; unique targets confirmed
    mean `4.25`; damaged **real RTB command yield 4/4**; deterministic meta-action mix
    `PLAN_COMPLIANCE 0 / OPPORTUNISTIC_ENGAGEMENT 18 / SELF_PRESERVATION_ABORT 13`.

  Held-out OVERALL mean moved `-0.6874998294268455` → `-0.2916663064230844`, **each over
  8/8 completed eval episodes**. Training: 6 successful episodes, **26 transitions**, **two
  productive iterations**, `updates_completed = 2`, train reward mean
  `-0.6874997530379319`, and `ended_counts` all `done` in both eval rounds. **These are
  SHORT-PROBE OBSERVATIONS, not estimates of converged policy performance**, and the two
  per-condition means are each over their own successful subset (§5).
  **ARTIFACT COMPLETENESS — reported ALONGSIDE the scientific denominators, never in place
  of one.** **24 visual-artifact bundles and 24 manifests**; **22 `complete`**, exactly
  matching the 22 successful attempts, and **2 `incomplete`**, exactly matching the two
  `setup` failures. Every `complete` bundle holds its known-only scenario, its executed
  t=0 scenario and its BLADE playback. Expected vs observed executed-world cardinality
  reconciles on every complete bundle at **3 known / 3 hidden / 6 total**. **Neither
  incomplete bundle fabricated a playback** (neither carries a recording at all). The
  complete scientific artifact remains preserved.
  **PLAYBACK WITNESSES AND CORROBORATING RUN EVIDENCE.** All of the following concerns the
  attempt preserved as
  `visual_artifacts/post_update_r001_e003_m1_seed1000003_damaged_tag901007`. **TWO evidence
  sources are involved and they are deliberately NOT merged.** (i) The BLADE **playback
  JSON** directly proves PHYSICAL state — position, fuel, `rtb`, route, weapon inventory,
  airbase membership. It is sampled every ten ticks (offsets below are stated from the
  recording's first frame) and it **does NOT record any per-wake meta-action label**;
  neither do `train_records.jsonl` / `eval_records.jsonl`, which persist per-round and
  per-iteration meta-action AGGREGATES only. (ii) The **preserved console transcript** of
  the run's per-episode `OK` blocks (`probe_console.log`, SHA-256
  `97bf45d56a3b224ef0ebe5a362bb7415b73e88520d192c891e347cb2412f31c4`, re-verified read-only
  before being cited) is the ONLY artifact that records a SELECTED ACTION LABEL, and it does
  so for the fuel-damage wake specifically.
  - **Defect A — KC-135R Stratotanker #76** (ego `0a14f756-13f2-4c78-8aa8-446da245aee5`, the
    id the playback binds to that name).
    *From the console transcript, for this exact attempt* — `[eval stage=post_update ep=3
    damaged seed=1000003] OK` records `fired=True tick=269 progress=0.300`,
    `fuel_before=203494.4 fuel_after=70026.7 factor=0.3441`, and
    `fd_wake=True fd_meta=SELF_PRESERVATION_ABORT rtb_command=True`. **The action label is
    an ATTRIBUTION FROM THAT RECORD, not something visible in playback.**
    *From the playback, independently* — the PHYSICAL signature at the same event: fuel
    falls from `203578.18` at the T+260 sample to `70017.43` at T+270 (sampled values, and
    therefore not identical to the console's exact event-time pair above), `rtb` flips
    `False → True` on that same frame, the route is replaced by a route to base at once, the
    aircraft lands at ≈ T+540, and it never resumes the abandoned assignment queue.
    The two sources agree, and the physical signature is **consistent with the merged
    EGO-GLOBAL abort semantics** rather than with removal of only the current assignment.
  - **Defect B — B-2 Spirit #698** (playback evidence alone; no action label is claimed).
    **ONE salvo per target**: one against Floridistan AFB #1794 at ≈ T+2320 (AIM-120
    `4 → 2`) and one against Hidden Airbase #003 at ≈ T+5140 (AIM-120 `2 → 0`). Each BLADE
    two-argument attack launches **two physical AIM-120 missiles**. No redundant second
    salvo against either target, and no repeated flat-timeout attack loop. Defect B changed
    NEITHER BLADE salvo quantity, NOR lethality, NOR general ammunition management — only
    the wait before a re-fire.
  - **Defect C — B-2 Spirit #698** (playback evidence, corroborated by the transcript's
    terminal line). It enters RTB at ≈ T+5240, **the episode keeps ticking while it
    physically flies home**, it lands at ≈ T+7705, and only then does the episode finish;
    the console block for the same attempt independently reports `ended=done ticks=7705
    dead=0`. That is physical completion, not RTB-command issuance.

  **DEFERRED RESEARCH HYPOTHESIS — NOT a defect, and NOT a proven action attribution.**
  Hidden Airbase #001 lies close to Hidden Airbase #003. In the sampled playback the B-2 is
  **50.07 km** from Hidden #001 at T+5230 while NOT yet in RTB — against a `DETECTION_KM`
  threshold of 50 km — and the next sampled frame, T+5240, shows RTB. Because playback is
  sampled every ten ticks and NOTHING preserved records an ORGANIC wake's selected
  meta-action — the jsonl records carry per-round / per-iteration AGGREGATES only, and the
  console transcript labels the FUEL-DAMAGE wake alone, which is a different ego in a
  different episode phase — it is PLAUSIBLE BUT NOT PROVEN that the B-2 crossed the threshold
  between samples and selected `SELF_PRESERVATION_ABORT` on the resulting wake. Treat
  possible over-conservatism as a FUTURE RESEARCH HYPOTHESIS about policy calibration —
  relevant to a later variable-FD-severity experiment. **Do not open a new defect, change
  the reward, retune the policy, or let this hypothesis invalidate or block this probe.**
  **EVIDENCE SHA-256** (verified read-only against the preserved run directory before being
  recorded here):
  `run_config.json=700f18fb54e485a12e0ab96a9b128353550c16c9f240e549bb40b01a303fbd22`,
  `train_records.jsonl=8581b4c50ad622ba2312c48434444cc57f560dfcd81c2495a03adf224666b16e`,
  `eval_records.jsonl=baa29c7281cf1dfbed9d928a83d60ab1e3c4826770de5ebcdf55ca34f12a68f2`,
  `episode_failures.jsonl=20c022a14971afb2776e774a268dbf0e6e6c0221fd2ac5e24dbe77e6c2f29784`,
  `run_summary.json=3038b754c82fb2dcb56d97632af4a24faa27dde68d2499701c228e3a208751fa`,
  `checkpoints/ckpt_iter0001.pt=605a05fde0084050fb66821e8da234bacacf1039a13f9bd0bd446876b9c2ba71`.
  **SUPERSEDING VERDICT (recorded with the roster-integrity lock below) — `INCONCLUSIVE —
  LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR`.** This run's
  own ledger records clean train seed 4 as an ACCOUNTED `setup` `EpisodeRosterError`. The
  approved roster-integrity correction (`36365f2`, below) establishes what that error
  actually was: a MEASUREMENT/DATA-INTEGRITY fault, which must ABORT the run — not an
  episode outcome that may quietly shrink a scientific denominator. So one of this probe's
  24 scheduled attempts was removed from the population by an instrument defect while the
  run reported itself reconciled, and a denominator produced that way cannot be read as
  sound. **CONSEQUENCES, stated exactly.** (i) The reward numbers, per-condition means,
  paired deltas, death counts, fuel-damage yield and PPO-productivity figures above are NO
  LONGER SCIENTIFIC EVIDENCE about the fuel-damage cell; they remain identifiable as raw
  historical outputs of this run and nothing more. (ii) The claim that this run PASSED or
  permanently released the long-baseline validity gate is WITHDRAWN. (iii) Everything
  factual is RETAINED and unchanged — the run identity, the one invocation and its exit
  code, the preset blob and `cli_overrides = []`, the complete provenance, the two elapsed
  quantities, the mechanical accounting, the 24/24 artifact bundles, the evidence hashes,
  and the three playback witnesses. (iv) The earlier review was NOT wrong about what it
  inspected; it was made against documentation that then described `EpisodeRosterError` as
  an accounted `setup` failure, so the fault presented itself as ordinary episode attrition.
  The verdict changed because that ROUTING was later found to be the defect. (v) This is
  **NOT** a fourth defect in Defects A, B or C — their corrections remain merged and
  witnessed; it is a SEPARATE roster/source-of-truth defect, closed by `36365f2`.

- **FIRST LONG BASELINE — EXECUTED / INDEPENDENTLY REVIEWED / `INCONCLUSIVE —
  ROSTER/DATA INTEGRITY FAILED`.** The engineering verdict was `REQUEST FIXES`. The run is
  attributable to exact code SHA `c30b6982ba605d60976cc303256da4b5528b0e63`
  (`2026-08-16T21:47:25+03:00`, the PR #23 merge), recorded Git branch
  `task/long-baseline-execution`, `dirty=false`, `dirty_path_count=0`, Windows /
  `nlp_env` (CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`. Per §7's
  hash convention this entry is keyed to the MEASURED CODE SHA. **No tracked file changed
  for the measurement** — it is a run of merged code, not a candidate.
  **RUN IDENTITY.** Run directory `training_output_long_baseline_100x8_seed0`. Exactly ONE
  invocation, native exit code **0**:
  `PYTHONPATH=src conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config training_output_long_baseline_100x8_seed0/long_baseline_contract.json`.
  `config_source.resolved_from = config_file`, `cli_overrides = []` — **no typed override
  and no ad-hoc knob**; `difficulty.factor = fuel_damage_baseline_v1`,
  `aircraft_penalty_coeff = 2.25`, `reward.formula_changed = false`. **Elapsed time is TWO
  DISTINCT QUANTITIES and they are never merged:** the harness's own
  `run_summary.json:run_seconds = 7764.3988857`, and the externally measured invocation
  wall clock of **7778.704310178757 s** (`timing.txt`:
  `PROBE_START_UTC = 2026-08-16T19:18:32.509409Z` →
  `PROBE_END_UTC = 2026-08-16T21:28:11.191698Z`). The harness figure excludes process
  start-up, `conda run` dispatch, imports and teardown, so it is NOT the wall clock.
  **THE SCIENTIFIC CONTRACT** (from the preserved `long_baseline_contract.json`, which is a
  MEASUREMENT contract and deliberately **not** a repository preset): 100 scheduled
  training iterations × 8 training attempts, train seeds `[0, 800)`; evaluation every 5
  iterations over 8 FIXED held-out seeds `[1000000, 1000008)`, each seed evaluated as
  `forced_clean` AND `forced_damaged`; **21 evaluation rounds** including the initial
  `pre_update`; the final 3-agent / 3-known / 3-hidden cell with its 200 km / 100 km
  geometry and `include_sams = false`; FD-BASELINE-v1 unchanged (`seeded_mixture`,
  `P(damaged) = 0.5`, leg progress `0.3`, RTB margin `1.10`); visual artifacts enabled for
  every scheduled attempt; `checkpoint_every = 10` → **10 checkpoints**.
  **MECHANICAL ACCOUNTING — historical fact, and NOT validity.** **1,136 scheduled
  attempts, 860 successful, 276 failed.** Training 800 attempted / 566 successful / 234
  failed; evaluation 336 attempted / 294 successful / 42 failed;
  `accounting_reconciled = true`; **100 productive iterations and 100 PPO updates**
  (`updates_completed = 100`). Every one of those counts reconciles, and **that is exactly
  the problem**: a run can be perfectly self-consistent about a population an instrument
  defect silently shrank, so these counts must never be offered as evidence that the
  measurement was sound.
  **FAILURE BREAKDOWN** (`failures_by_pipeline_stage` = `{"setup": 276}`;
  `failures_by_error_type` = `{"RuntimeError": 101, "EpisodeRosterError": 143,
  "FuelDamageError": 32}`; `failures_by_condition` = `{"clean": 123, "damaged": 153}`):
  - **143 `EpisodeRosterError`, ALL in training, spread over 83 distinct iterations** — 75
    clean and 68 damaged. Two shapes: **126 PRE-run** roster failures claiming a t=0 known
    target was absent from the executed world (125 naming one target, 1 naming two), and
    **17 POST-run** failures raised after a real episode and a real playback because a
    CONFIRMED target id fell outside the incorrectly shortened roster.
  - **101 B2 exact-cardinality `RuntimeError`** — 59 train, 42 eval.
  - **32 `FuelDamageError`**, every one a DAMAGED TRAINING attempt.
  **INDEPENDENT ARTIFACT REVIEW — what falsified the run.** Every one of the 126 pre-run
  roster failures had a FULL SIX-TARGET authoritative `executed_t0_scenario.json`; the 17
  post-run failures left real playback files their `incomplete` manifests did not list; and
  **11 `complete` manifests reported observed `3 known / 2 hidden / 5 total` while their own
  authoritative executed-t0 scenarios held `3 + 3 = 6`** (re-verified here across all 1,136
  preserved manifests: 860 `complete`, 276 `incomplete`, and exactly 11 of the complete ones
  carrying that `5`-total observation). **ROOT CAUSE: allocated-only solver output was being
  read as world inventory** — closed by `36365f2` below.
  **VERDICT: engineering `REQUEST FIXES`; scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY
  FAILED`.** **Do NOT report this run's reward improvement, per-condition means, paired
  deltas, survival, fuel-damage yield or PPO performance as scientific evidence.** Those
  values are present in the preserved records and may be referred to ONLY as raw historical
  outputs of an inconclusive run; they are deliberately not tabulated here, precisely so
  they cannot be lifted out of context as a baseline. **The 101 B2 and 32 fuel-window
  failures are NOT corrected by `36365f2`** — they remain EXPECTED scientific outcomes under
  the current contract (§8) and must not be relaxed, retried, retuned or reclassified.
  **PRESERVATION.** The run directory is preserved and must not be modified, moved, copied,
  repackaged, deleted or regenerated. **EVIDENCE SHA-256** (verified read-only against the
  preserved run directory before being recorded here):
  `long_baseline_contract.json=18d0dede02b8b89cfff8867aefdd68901d995f1664dcbba7342e26a9bbed02ac`,
  `run_config.json=fae72de5f7c10ec5c9264330510d9ab9fac8af34c5270f1912a0f4d36b9526e2`,
  `run_summary.json=6ea6842ed981219c7dd45fb9cbc63587a7e4c26d57a7602c7f6676aaa31d2848`,
  `train_records.jsonl=d9d94f9a18448565a31a45c2ec950d1882e7b130e3cd16a181acc1074b4aa96c`,
  `eval_records.jsonl=e378da7ff7c3cb21d63ca016fa5d0911fe6209a87d757467b89a64fc043b7edb`,
  `episode_failures.jsonl=dcf7871e16102a5b9d090a16276603d47611763fdfbb125c584978edbd3cac32`,
  `timing.txt=828d5c0d390c2e082ca4d22c652c8d36b51b23c5140c29139651897f8b8fa10a`,
  `long_baseline_console.log=00d35661b6246091d1e199944c90571233606f76c8d571287215d9b218b60233`.
  The review package additionally carried
  `review_metadata/package_manifest.json=a5b7acd5d607958e54b81e8ba2f354155f18c19a9803432bbbee1f0e9fbfb2c4`
  and
  `review_metadata/playback_audit.jsonl=4c5bbb1c9cd629c20dde761137f8b2fd9e83ff2b5b4a398410b2abf7c7242137`,
  inside a review ZIP whose own SHA-256 is
  `b22aecec7b1c99d3689ce6ad34d8c467473bb7a50d0d3bd25a3a5f4c370440af`; those two live in the
  review package rather than in the preserved run directory, so they are recorded from the
  review record rather than re-derived here.

- **PHASE-A LONG BASELINE (RERUN) — EXECUTED / INDEPENDENTLY REVIEWED / `APPROVE — VALID
  MEASUREMENT`. THE FIRST SCIENTIFICALLY VALID MEASUREMENT OF THE FUEL-DAMAGE CELL.** The
  measurement is attributable to exact clean code SHA
  `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` (committed `2026-08-17 19:25:42 Asia/Jerusalem`),
  the `main` head produced by the roster-integrity documentation merge (PR #25). Per §7's
  hash convention this entry is keyed to the MEASURED CODE SHA; the documentation commit that
  creates it, and the merge that integrates it, cannot name their own SHAs and are
  deliberately not invented here. **No tracked file changed for the measurement** — it is a
  run of already-reviewed merged code, not a candidate.
  **RUN IDENTITY.** Run directory
  `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`. Exactly ONE invocation,
  native exit code **0**, `PYTHONPATH=src`, from the repository root:
  `conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf/long_baseline_contract.json`.
  `config_source.resolved_from = config_file` and **`cli_overrides = []`** — the reviewed
  measurement contract with NO typed override and NO ad-hoc knob. Provenance complete:
  `git.available=true`, exact SHA, `branch` resolved, `dirty=false`, `dirty_path_count=0`,
  Windows / `nlp_env` (CPython 3.12.3), torch 2.7.1+cpu, gymnasium 1.0.0, vendored BLADE,
  BONMIN available; `difficulty.factor = fuel_damage_baseline_v1` with
  `aircraft_penalty_coeff = 2.25` and `reward.formula_changed = false`. **Elapsed time is TWO
  DISTINCT QUANTITIES and they are never merged:** the harness's own
  `run_summary.json:run_seconds = 8493.042731400012`, and the externally measured invocation
  wall clock of **8509.632915974 s** (`timing.txt`:
  `PROBE_START_UTC = 2026-08-17T22:32:34.530300100Z` →
  `PROBE_END_UTC = 2026-08-18T00:54:24.166136300Z`). The harness figure excludes process
  start-up, `conda run` dispatch, imports and teardown, so it is NOT the wall clock.
  **CONTRACT FIDELITY.** The scientific contract is the preserved authoritative
  `long_baseline_contract.json` of the invalid first long baseline (SHA-256
  `18d0dede…02ac`), cloned with **exactly ONE field changed — `output_dir`**. Verified before
  execution: 27 keys, identical key SETS and identical key ORDER, exactly one differing key,
  every other value identical. So the train seeds `[0, 800)`, the held-out seeds
  `[1000000, 1000008)`, the 100 × 8 schedule, `eval_every = 5`, the matched
  forced-clean / forced-damaged pair design, the PPO settings, the 3-agent / 3-known /
  3-hidden cell with its 200 km / 100 km geometry, `include_sams = false` and every
  FD-BASELINE-v1 parameter are UNCHANGED. **A directory name is not a scientific parameter.**
  **ACCOUNTING — every denominator explicit, `skip_and_account_v1` unchanged.**
  **1,136 scheduled attempts, 993 successful, 143 failed**, `accounting_reconciled = true`.
  Training 800 attempted / 699 successful / 101 failed, every success wake-bearing;
  evaluation 336 attempted / 294 successful / 42 failed across **21 rounds**. **ZERO
  `EpisodeRosterError`, ZERO `MeasurementIntegrityError`, ZERO `_VisualArtifactError`, and
  zero crash outside the `generation`/`setup`/`run`/`reward` episode taxonomy** — every one
  of the 143 failures is an accounted episode failure, each recorded exactly once, with no
  retry, no substitution and no band shift.
  **FAILURES — both families are EXPECTED SCIENTIFIC OUTCOMES, not defects.** Exactly **101
  B2 exact-cardinality `RuntimeError`** and exactly **42 `FuelDamageError`**, all at stage
  `setup`. The independent review established two exact set relations: the 101 B2 failures
  are **exactly the same scheduled attempts as in the invalid old long run**, and
  `EpisodeRosterError` went **143 (old) → 0 (new)**, with the **10 additional
  `FuelDamageError` attempts relative to the old run being exactly attempts that were
  `EpisodeRosterError` there**. Held-out seed **`1000005`** fails B2 for BOTH matched members
  in ALL 21 evaluation rounds, so the matched-pair yield is a STRUCTURAL **7/8 in every
  round** — a property of that seed's world, not stochastic attrition.
  **LEARNING — reported only after the validity gate passed.** 100 scheduled iterations gave
  **100/100 productive PPO updates** (`updates_completed = 100`) over **2,566 transitions**,
  with 0 zero-wake and 0 all-failed iterations. The matched held-out paired reward delta
  improved from **−0.375000 over 7/8 pairs** at `pre_update` to **−0.071429 over 7/8 pairs**
  at the final `post_update`, and evaluation aircraft deaths fell from **7 to 0**. Final
  clean reward reached **0 on all 7** exact-cardinality-feasible held-out worlds; final
  damaged reward reached **0 on 5 of those 7**, with the residual damaged cost concentrated
  in exactly two worlds — seed **1000004** (`−0.333333`, 4/6 targets, 0 deaths) and seed
  **1000007** (`−0.166667`, 5/6 targets, 0 deaths), i.e. both PRESERVED THE AIRCRAFT at the
  cost of incomplete target coverage. In all **7** completed damaged held-out worlds the
  deterministic fuel-damage decision changed from `PLAN_COMPLIANCE` before training to
  `SELF_PRESERVATION_ABORT` after training, and selected playback witnesses independently
  confirm this as a REAL PHYSICAL behavioural change including survival and RTB under the
  final policy. **The two per-condition means are each over their own successful subset**, so
  the within-seed claim is the matched-pair delta alone (§5).
  **FUEL-EXPOSURE CAVEAT — state it against the right denominator.** Successful damaged
  TRAINING episodes number **324** while damage events actually fired **323** times; the one
  non-firing successful damaged training episode is **seed 424** (iteration 53), whose
  selected ego returned before reaching the 0.30 leg-progress trigger. **No defect is
  inferred** — a fuel-damage PLAN existed and every LIVE quantity is recorded as `n/a`, so
  the artifacts record that the event did not fire and do not record why. **Evaluation
  exposure is COMPLETE: 147 / 147** successful damaged eval episodes fired and woke their
  selected ego.
  **ARTIFACT COMPLETENESS — reported ALONGSIDE the scientific denominators, never in place of
  one.** 1,136 bundles and 1,136 manifests: **993 `complete`** (exactly the successful
  attempts) and **143 `incomplete`** (exactly the failed ones). **All 993 complete bundles
  reconcile expected against observed 3 known / 3 hidden / 6 executed targets** — the
  five-target contradiction that falsified the old run does not occur once. No incomplete
  bundle fabricated a playback, and no completed run left an unlisted one.
  **INDEPENDENT REVIEW.** A read-only evidence package
  `long_baseline_rerun_737b4bf_gpt_review.zip` (SHA-256
  `f2582c0ca7f460a5f51bd515aeb0506f0476e8e06e4039312b7371858a08b932`) carried the core
  evidence of both runs, all 1,136 original manifests, selected raw playback bundles and
  derived audits. The GPT verdict is **`APPROVE — VALID MEASUREMENT`**.
  **EVIDENCE SHA-256** (verified read-only against the preserved run directory):
  `long_baseline_contract.json=f5b5984317ea503862fcf76670bac0f4c3f147f39d8daf969ab90009ff438c1f`,
  `run_config.json=eeb4f449ead84b5cf7a72c6248a810169e8eb5f36fa7ba94384cab8a9bd1fb4a`,
  `run_summary.json=ee32e8b7b6735351700d19fc560c840307c19c0af7a446525e09dc586154b71d`,
  `train_records.jsonl=29c2a40bce2267af5aff60d281258e27eadd432db2291f3a2d0f29854a4cc1bd`,
  `eval_records.jsonl=116022680a7d8df97466c7c43faa7b6ff5b3b783403709f5d285bca66ca1995f`,
  `episode_failures.jsonl=313990a1428d8bde71c25db6bbc55b33a731ef942d0dff994e1e06b16a4b6ea1`,
  `timing.txt=27a11d9867f88dc04ec6f5d9d1ff75c1fc51aefbaa5bdc9ddf61fbd3b92b4a9c`,
  `long_baseline_rerun_console.log=be3c97d3106b8d18523d433e60e98dd06a1d98f00051d9752b9b01d25deff9a6`.
  **PRESERVATION.** This run directory is preserved and must not be modified, moved, copied,
  repackaged, deleted or regenerated — and neither may the invalid
  `training_output_long_baseline_100x8_seed0`, which remains preserved and explicitly
  scientifically INCONCLUSIVE.
  **PHASE-A SCIENTIFIC CONCLUSION — VALID BASELINE.** The first scientifically valid
  long-baseline measurement of FD-BASELINE-v1 was obtained from the clean actor-only,
  no-communication graph-RL stack at measured code SHA
  `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`. Across the seven exact-cardinality-feasible
  held-out matched worlds, the paired fuel-damage reward penalty improved from `−0.375000`
  before training to `−0.071429` after 100 productive PPO updates, while evaluation aircraft
  deaths fell from 7 to 0. In all seven completed damaged held-out worlds the deterministic
  policy changed from `PLAN_COMPLIANCE` before training to `SELF_PRESERVATION_ABORT` after
  training. Final clean performance reached reward 0 on all seven feasible worlds; final
  damaged performance reached reward 0 on five, while the remaining two preserved the
  aircraft at the cost of incomplete target coverage. These results establish **end-to-end
  learnability and meaningful ego-local runtime adaptation in the locked Phase-A reference
  cell.** **THE EXPLICIT NON-CLAIMS:** they do **NOT** establish global optimality, **NOT**
  monotonic convergence, **NOT** generalization beyond this fixed cell and this held-out seed
  set, and **NOT** any benefit from centralized training. Those are subsequent research
  questions. **PHASE A IS CLOSED BY THIS ENTRY.**

- **FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE — EXECUTED / INDEPENDENTLY REVIEWED /
  `APPROVE — VALID MEASUREMENT`. THE FIRST AND ONLY SCIENTIFICALLY VALID MEASUREMENT OF THE
  VARIABLE-SEVERITY CELL, AND ITS PRIMARY BEHAVIOURAL FINDING IS NEGATIVE.** The measurement
  is attributable to exact clean code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
  `dd881478b8e2e521054d09bc865437f1308be1a2` (committed
  `2026-08-20 14:50:24 Asia/Jerusalem`, the `main` head produced by the variable-severity
  documentation merge, PR #28). Per §7's hash convention this entry is keyed to the MEASURED
  CODE SHA; the documentation commit that creates it, and the merge that integrates it,
  cannot name their own SHAs and are deliberately not invented here. **No tracked file
  changed for the measurement** — it is a run of already-reviewed merged code, not a
  candidate. It was executed from a clean DETACHED snapshot at that SHA
  (`provenance.git.repo_root` = `…\fd_variable_severity_v1_bf1e045f_snapshot`,
  `branch = HEAD`, `dirty = false`, `dirty_path_count = 0`), so repository work landing
  after `bf1e045f…` — Phase-B CTDE included — is outside the measured tree and neither
  contaminated the run nor is attributable to it.
  **RUN IDENTITY.** External measurement root `C:\Users\Itama\f7r2`; contract `c.json`;
  output directory `r`; console `console.log`; timing `timing.json`. Exactly ONE
  invocation, native exit code **0**:
  `conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config "C:\Users\Itama\f7r2\c.json"`.
  `config_source.resolved_from = config_file` and **`cli_overrides = []`** — the reviewed
  measurement contract with NO typed override and NO ad-hoc knob. Provenance complete:
  `git.available = true`, exact SHA, clean, Windows 10.0.19045 / `nlp_env`
  (CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`;
  `difficulty.factor = fuel_damage_variable_severity_v1` with
  `fuel_damage_mode = seeded_variable`, scheduled cell probabilities
  `clean 0.50 / mild 0.25 / severe 0.25`, `target_policy = live_severity_midpoint_v1`,
  `aircraft_penalty_coeff = 2.25` and `reward.formula_changed = false`. **Elapsed time is
  TWO DISTINCT QUANTITIES and they are never merged:** the harness's own
  `run_summary.json:run_seconds = 5998.791282300022`, and the externally measured
  invocation wall clock of **6021.3954213 s** (`timing.json`:
  `start_utc = 2026-08-22T14:36:14.1838098Z` → `end_utc = 2026-08-22T16:16:35.5973421Z`).
  The harness figure excludes process start-up, `conda run` dispatch, imports and teardown,
  so it is NOT the wall clock.
  **CONTRACT FIDELITY — a REPLACEMENT, not a redesign.** The scientific contract is the
  invalid precursor's own contract cloned with **exactly ONE field changed —
  `output_dir`**. Verified read-only before this record: 25 keys, identical key SETS and
  identical key ORDER, exactly one differing key, every other value identical. So the
  §8-approved run shape is unchanged: **50 scheduled training iterations × 8 scheduled
  training attempts = 400**, `base_seed = 0`; evaluation every 5 iterations INCLUDING the
  initial `pre_update` ⇒ **11 evaluation rounds**; **8 fixed held-out seeds** from
  `1_000_000`, each a matched **clean / mild / severe TRIAD** ⇒ **11 × 8 × 3 = 264**
  scheduled evaluation attempts; **664 scheduled attempts in total; NO early stopping**;
  the locked cell of 3 agents / 3 known / 3 hidden with its 200 km / 100 km geometry,
  `DETECTION_KM = 50`, `include_sams = false`, target-destruction `probability = 1`, frozen
  solver and BLADE, unchanged `graph_reward` formula and unchanged actor-only PPO;
  `visual_artifacts = true`. **A directory name is not a scientific parameter.**
  **THE EXCLUDED PRECURSOR — `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT`.** An earlier
  attempt at the SAME contract, preserved at
  `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640\training_output_fd_variable_severity_v1_50x8_seed0`,
  is **preserved historical / ENGINEERING evidence ONLY and is EXCLUDED from every
  scientific reading.** Its ledger carries the same 78 `setup` failures (58 B2
  `RuntimeError` + 20 `FuelDamageError`) **plus 70 additional `run`-stage
  `FileNotFoundError`s, and all 70 are `post_update` SEVERE evaluation members** — 10
  post_update rounds × the 7 exact-cardinality-feasible held-out seeds, i.e. **the entire
  post-training severe arm, which is precisely the arm the experiment exists to measure.**
  The cause is a Windows `MAX_PATH` playback-export failure: the BLADE recording path was
  **267 characters** against the 260-character limit. **This is NOT a negative scientific
  result** — it is an infrastructure failure that systematically deleted one experimental
  cell, and a population destroyed that way yields no result rather than a null one. The
  precursor is preserved and must not be modified, moved, copied, repackaged, deleted or
  regenerated. Its evidence SHA-256 (re-verified read-only before this record):
  `fd_variable_severity_v1_contract.json=77e5992994235abd0962547b549dbfb17889cb51f745993f4c8f2a89a2824326`,
  `invocation_timing.json=cf531cd2f0574674d66d21cb16e12d963aeb269fac9d9ae55cdaebf1a47c26b6`,
  `run_config.json=de20663b689f0a00f483c30ac92e6b240c5df2873d5d3f0aa9145b550f71ea1e`,
  `episode_outcomes.jsonl=bd0f1c4009cec2ea596db372f3d0e76c41c8e0a2352aa3775b108fcad2e544c6`,
  `episode_failures.jsonl=430728726b90383a47e1f8f62b997ed82a12ea0f23c205e88b8a7a360a536e0b`,
  `run_summary.json=79b30564da6f8157de536f5678dbd67851516b0baa20539034929b4f7b6f0e85`.
  **ACCOUNTING — every denominator explicit, `skip_and_account_v1` unchanged.**
  **664 scheduled attempts, 586 successful, 78 failed**, `accounting_reconciled = true`,
  and `586 + 78 = 664`. Training **400 attempted / 355 successful / 45 failed**, every
  success wake-bearing; evaluation **264 attempted / 231 successful / 33 failed** across
  **11 rounds**. Per CELL — training `clean 202 / 190 / 12`, `mild 92 / 76 / 16`,
  `severe 106 / 89 / 17`; evaluation `clean 88 / 77 / 11`, `mild 88 / 77 / 11`,
  `severe 88 / 77 / 11`. **Every one of the 78 failures is at stage `setup`**: **58 B2
  exact-cardinality `RuntimeError`** and **20 `FuelDamageError`** (no valid strict fuel
  band/window). All **33 evaluation failures are held-out seed `1000005`** — 11 rounds × 3
  triad members — the same structural B2 world the Phase-A baseline also lost, reported and
  never repaired. **ZERO `FileNotFoundError`, ZERO `MeasurementIntegrityError`, ZERO
  `EpisodeRosterError`, ZERO `_VisualArtifactError`, and zero crash outside the
  `generation`/`setup`/`run`/`reward` episode taxonomy.** Outcome and failure identities
  are unique and DISJOINT (586 records in `episode_outcomes.jsonl`, 78 in
  `episode_failures.jsonl`, zero overlap), and no scheduled-vs-executed CELL mismatch abort
  occurred, so every successful attempt was booked in the cell the schedule asked for.
  **MATCHED TRIADS — a structural 7/8 in EVERY round.** All 11 evaluation rounds report
  **7/8 complete clean+mild+severe triads**, including `pre_update`
  (`updates_completed = 0`) and the final `post_update` (`updates_completed = 50`). Seed
  `1000005`'s B2 failure is the ceiling; it is a property of that world, not stochastic
  attrition.
  **PRIMARY BEHAVIOURAL RESULT — NO SEVERITY-CONDITIONED META-ACTION SEPARATION. THIS IS
  THE FINDING, AND IT IS NEGATIVE.** Rates are over **FD WAKES**, never over episodes.
  - **`pre_update`** — MILD: 7 wakes, `PLAN_COMPLIANCE 7/7`, abort 0, engage 0.
    SEVERE: 7 wakes, `PLAN_COMPLIANCE 7/7`, abort 0, engage 0.
  - **FINAL `post_update` (`updates_completed = 50`)** — MILD: 7 wakes,
    `PLAN_COMPLIANCE 7/7`. SEVERE: 7 wakes, `PLAN_COMPLIANCE 7/7`.
  - **ALL TEN `post_update` ROUNDS COMBINED** — MILD: 70 wakes,
    `PLAN_COMPLIANCE 63 = 0.900`, `SELF_PRESERVATION_ABORT 7 = 0.100`, engage 0.
    SEVERE: 70 wakes, `PLAN_COMPLIANCE 63 = 0.900`, `SELF_PRESERVATION_ABORT 7 = 0.100`,
    engage 0. **The two distributions are IDENTICAL.**
  - **TRAINING successes** (stochastic policy, context only) — MILD: 76 wakes,
    `PLAN_COMPLIANCE 60`, `SELF_PRESERVATION_ABORT 16`. SEVERE: 89 wakes,
    `PLAN_COMPLIANCE 66`, `SELF_PRESERVATION_ABORT 23`.

  The deterministic held-out actor did NOT differentiate a survivable MILD fuel loss from
  an unsurvivable SEVERE one in its FD-wake action choice, before OR after training. **This
  is a VALID NEGATIVE SCIENTIFIC RESULT.** It does **NOT** mean the actor is broken, that
  training failed, that PPO did not learn, that the actor never uses fuel at all, or that
  the result generalizes beyond this fixed cell and this held-out seed band. And **"mild
  must choose `PLAN_COMPLIANCE`" is NOT a correctness rule** (§5) — what was measured is
  whether the response DIFFERS, not whether it matched a prescribed label.
  **DENOMINATOR / INDEPENDENCE CAVEAT — load-bearing.** The clean statistical unit for the
  FINAL held-out policy is the final round's **7 complete matched triads**. The 70
  `post_update` observations per severity REUSE those same seven feasible held-out seeds
  across ten checkpoints; they describe the learning TRAJECTORY across checkpoints, and
  **they are NOT 70 independent held-out worlds and must never be used to inflate sample
  size.**
  **PHYSICAL OUTCOMES — THE SEVERITY FACTOR IS REAL.** The absence of behavioural
  separation is NOT because the two cases are physically equivalent. Over the successful
  `post_update` evaluation outcomes: **CLEAN** 70 episodes, 0 RTB commands, 0 deaths, mean
  unique target coverage **6.000 / 6**; **MILD** 70 episodes, **70** RTB commands, 0
  deaths, **5.957 / 6**; **SEVERE** 70 episodes, **43** RTB commands, **63** deaths,
  **5.700 / 6**. At the FINAL round all seven feasible clean worlds and all seven feasible
  mild worlds reach 6/6 with 0 deaths, while **every one of the seven feasible severe
  worlds loses one airframe** — five at reward ≈ `−0.375` with 6/6 coverage, and seeds
  **1000004** and **1000007** at ≈ `−0.541666` with 5/6 coverage. RTB yield is real Phase-2
  COMMAND HISTORY (`FuelDamageOutcome.rtb_command_issued`), never the executor's
  `rtb_issued` latch (§5).
  **REWARD AND THE THREE WITHIN-SEED DELTAS — over COMPLETE TRIADS ONLY.** Per-cell means
  are each over THAT cell's own successful subset, so subtracting two of them is not a
  matched effect; the only within-seed claims are the deltas below, each over `n = 7`
  complete triads.
  - `pre_update` (n = 7): clean `-0.49999970`, mild `-0.49999970`, severe `-0.87499991`;
    `mild − clean = 0.0`, `severe − clean = -0.37500021`, `severe − mild = -0.37500021`.
  - final `post_update` (n = 7): clean `5.714293e-07`, mild `5.714293e-07`, severe
    `-0.42261872`; `mild − clean = 0.0`, `severe − clean = -0.42261929`,
    `severe − mild = -0.42261929`.

  **PPO PRODUCTIVITY — training WAS productive.** **50 / 50 scheduled training iterations
  productive**, 0 zero-wake, 0 all-failed, `n_epochs_run = 4` in every iteration,
  `updates_completed = 50`, **1,405 transitions**. Training reward improved from
  `-0.51547582` to `-0.07738026`. **The correct reading is therefore precise:** actor-only
  PPO training was productive and improved overall performance, and it nevertheless did NOT
  produce the targeted held-out MILD-vs-SEVERE behavioural differentiation. **No claim that
  CTDE would fix this is made or supported here** — no CTDE benefit is measured by this run.
  **ARTIFACT COMPLETENESS — reported ALONGSIDE the scientific denominators, never in place
  of one.** **664 bundles and 664 manifests: 586 `complete`** (exactly the successful
  attempts) and **78 `incomplete`** (exactly the failed ones), **586 playbacks**, 2,520
  files, ≈ **4,430.6 MB**. **All 586 complete bundles reconcile expected against observed
  3 known / 3 hidden / 6 executed targets.** No incomplete bundle fabricated a playback,
  and no completed run left an unlisted one. **No path-related artifact failure occurred:
  the maximum actual artifact/playback path is 139 characters**, because this run
  deliberately used the short external root after the precursor proved the `MAX_PATH`
  hazard.
  **ENGINEERING CAVEAT — HISTORICAL FACT, NOT A CODE CHANGE AND NOT FIXED HERE.** The
  precursor proved that a BLADE playback-export failure can currently surface as an
  ORDINARY `run`-stage `EpisodeAttemptError` and therefore enter the SCIENTIFIC failure
  ledger: `graph_tick_loop.run_episode`'s `ctx.game.export_recording()` propagates into
  `graph_train._run_one_episode`, which does `raise EpisodeAttemptError("run", exc) from exc`.
  An artifact/serialization fault raised through `_VisualArtifactError` is INFRASTRUCTURE
  and aborts the run (§5); this one is not routed that way and was accounted as ordinary
  episode attrition. **The valid run had ZERO such failures**, so nothing about this
  measurement depends on it. It is recorded as measurement-infrastructure history and a
  future engineering caveat; **no code was changed for it in this record**, and changing
  that routing would be its own separately reviewed task.
  **INDEPENDENT REVIEW.** The GPT orchestrator independently reviewed the replacement
  measurement and issued **`APPROVE — VALID MEASUREMENT`**. The precursor's verdict is
  **`INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT`**.
  **EVIDENCE SHA-256** (re-verified read-only against the preserved measurement root before
  being recorded here):
  `c.json=a3961058cc36b2e1b83199e87d0799d3d8042e4cd1966930328f40c451e0fa02`,
  `timing.json=cc4cab1dd718b80712b8d3e79ed2ecb453ac1a9eedbfa711277594cf6f7d96e0`,
  `console.log=6618acb98b0b77439ed2e494c0705df4cb4033f37ac767e9b7044084978435fd`,
  `r/run_config.json=3e43e54b7f3b0685e4770e3c75e6a57a523147460c2785d31c7f470b62202375`,
  `r/train_records.jsonl=a79b54e9c15c169872968a6b63f1699f4c9c7502d6cf61054f311317d1bea806`,
  `r/eval_records.jsonl=4a0d867872901bc778673ed574487abca600c0afdd3c70f64d17f3909c951aa5`,
  `r/episode_outcomes.jsonl=e47a4c35ece4349f870b961d42fe3a29b44cd4f064f793e675c022a7cb240239`,
  `r/episode_failures.jsonl=88d7869881b137696a2a4e8e921f8a2537db651165ec8183bb6990be3c5305e3`,
  `r/run_summary.json=f0ced3cd612b22f1160ff7bda38a2884df98b7fb559ffef44af737d0e1cf4d5e`,
  `r/plots/training_performance.png=86b5b689cbdd98a9cb271746539a0f90cbe52d25f4353ca6619d557e5191fd71`,
  `r/plots/policy_diagnostics.png=a9b24a239320fb17b8d9aad139d63a5fb1a317031c8557a6744bb2a10ef943ef`,
  `r/plots/measurement_health.png=76c6af6a56619701d09ed52fe02ae4fc0fc12f80fbbf36157a2214947ab8bbb1`.
  `run_summary.json:/severity_response` is DERIVED from `episode_outcomes.jsonl`
  (`severity_response_source`), which is the ONE metric path (§5).
  **PRESERVATION.** BOTH measurement trees — the VALID run root `C:\Users\Itama\f7r2` and
  the INVALID precursor
  `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640` — are preserved and must
  not be modified, moved, copied, repackaged, deleted or regenerated, and neither may any
  earlier preserved run.
  **SCIENTIFIC CONCLUSION.** A valid actor-only baseline of FD-VARIABLE-SEVERITY-v1 was
  obtained from the clean, no-communication graph-RL stack at measured code SHA
  `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`. The severity construction is PHYSICALLY REAL
  in this cell — mild and severe diverge sharply in RTB yield, airframe survival and target
  coverage — and actor-only PPO training was productive across 50/50 updates, improving
  training reward from `-0.51547582` to `-0.07738026`. **Nevertheless the deterministic
  held-out actor showed NO severity-conditioned FD-wake meta-action separation**: at
  `pre_update` and at the final `post_update` it chose `PLAN_COMPLIANCE` in all 7 completed
  MILD and all 7 completed SEVERE matched worlds, and across all ten `post_update`
  checkpoints the two per-severity distributions are identical. **THE EXPLICIT
  NON-CLAIMS:** this does **NOT** establish that the actor is broken, that training or PPO
  failed, that the actor ignores fuel entirely, that MILD "should" have chosen
  `PLAN_COMPLIANCE`, that 70 post-update observations per severity are 70 independent
  held-out worlds, that the finding generalizes beyond this fixed cell and this held-out
  seed band, or that centralized training would change it. Those are subsequent research
  questions, and **this is a valid negative result — not a defect, and not grounds for
  retuning, re-seeding or re-running.**

- **GENERALIZED-V1 TASK 5A / TASK 5B — BOUNDED ENGINEERING VALIDATION. THE LABEL IS
  BINDING: ENGINEERING VALIDATION, NOT SCIENTIFIC MEASUREMENT.** Both were independently
  reviewed `APPROVE — VALID ENGINEERING VALIDATION`.
  **WHAT MAKES THEM NOT MEASUREMENTS IS THEIR DESIGNATED PURPOSE, NOT AN ABSENCE OF
  MECHANICS.** Task 5B in particular really did carry engineering mechanics — an explicit
  training seed band `[720000, 720072)`, an explicit benchmark candidate band, the
  PRODUCTION held-out verification, a TRANSIENT frozen manifest, and 18 worlds / 54 members
  for its ONE evaluation round. **Those mechanics existed solely to validate system
  behaviour, attrition and runtime, and were explicitly NOT designated as the scientific
  comparator or as a policy-performance measurement.** Both runs were AUTHORIZED, EXECUTED
  and REVIEWED as engineering validation, and that designation is what the label rests on.
  Therefore, and bindingly: **no reward or learning claim, no generalized-performance claim,
  no actor-vs-CTDE claim, and no promotion of Task 5B's transient manifest into R1.** They
  are recorded here for exactly one purpose — they are what the currently authorized
  campaign's mechanics were chosen from — and this DOCUMENTATION task re-ran neither of them
  and reports their findings as ALREADY-REVIEWED evidence rather than as anything it
  verified itself. *(SUPERSEDED, and corrected here: this entry previously said neither run
  had a scientific contract, a seed schedule, a held-out band, a frozen comparator or a
  population denominator. That is too broad and is factually wrong for Task 5B, which had
  seed bands, held-out verification and a transient frozen manifest; the engineering-only
  label is UNCHANGED and rests on designated purpose instead.)*
  - **TASK 5A — the bounded end-to-end generalized rehearsal.** It ran against a TRANSIENT
    one-world-per-cell benchmark, and its load-bearing engineering finding is the one that
    changed the design: an A2-LOW world failed REPEATEDLY on `pre_event_popup_risk`, the
    certified-FD eligibility rejection reason, which is what exposed the need for
    ELIGIBILITY SELECTION BEFORE THE FREEZE — a world frozen unchecked fails the SAME
    member in every validation round of every arm, forever. That is the defect
    `graph_benchmark_preflight` exists to close (§5). Solver runtime DOMINATED execution.
    **A CAVEAT THAT IS PART OF THE FINDING: repeated pre-update and post-update values
    measured on the SAME world are NOT independent runtime observations**, and must never be
    counted as such. **No reward and no policy behaviour from Task 5A is promoted anywhere.**
  - **TASK 5B — bounded mechanism validation at measured code SHA
    `4af6c5aa5dd28072692bfda63282964b55010aae`.** The validated engineering facts, recorded
    only because later execution planning depends on them: generalized training completed
    **24/24 successful attempts with 0 ordinary failures**; the benchmark preflight accepted
    **18/18 FIRST candidates with 0 rejections**; **0 requested-to-realized hidden shortfalls
    were observed** in those bounded samples; one TRANSIENT benchmark round completed
    **54/54 members successful with 18/18 complete matched groups**; the real BONMIN
    reference solver DOMINATED runtime; **`A4-high` showed very large runtime variance**; and
    **one legitimate training solve of roughly 998 seconds terminated `optimal`** — which is
    precisely why **NO short solver timeout was adopted**, since a timeout below that would
    have killed a solve that was going to answer correctly.
    **THE SAMPLE-SIZE LIMITATION IS EXPLICIT AND BINDING.** These are BOUNDED samples. **No
    attrition-rate population claim may be made from them** — "0 rejections in 18 first
    candidates" and "0 shortfalls" are observations about those samples and are NOT an
    estimate of the rate a full campaign will see, which is exactly why
    `generalized_max_attempts_per_iteration` and `max_candidates_per_cell` are REQUIRED
    operator inputs with no defaults (§5). **No learning claim, no actor-vs-CTDE claim and
    no final scientific result is established by Task 5B**, and its transient benchmark is
    **NOT** the R1 comparator.

- `88352b2f` — **GENERALIZED-V1 TASK 5 DOCUMENTATION / LOCK CHECKPOINT — CLOSED / APPROVED /
  MERGED.** FINAL approved documentation candidate SHA
  `88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` (committed `2026-08-30 23:51:07 +0300`), on
  branch `task/generalized-v1-task5-doc-lock`, integrated by merge commit
  `9b9e9b85a70c8a0019c72ada92ceec3401725795` (`2026-08-31 00:39:29 +0300`, **PR #44**).
  Grade A under `GPT_GITHUB`, implementation mode SURGICAL — Grade A because it locks
  research-validity contracts and phase state, not because it changes behaviour. **IT IS A
  DOCUMENTATION LOCK: it changed NO source, test, config, preset, benchmark manifest, run
  artifact or scientific result.**
  **REVIEWED SCOPE: EXACTLY TWO FILES** — `CLAUDE.md` and `graph_rl_project_handoff.md`
  (verified as the complete `b3c2e01f…...9b9e9b85…` comparison). It recorded the Task-5 §5
  contract, its §6 routing rows, the §7 entries for PR #42 and PR #43, the Task-5A /
  Task-5B engineering-validation entry under its BINDING label, and the dispatched
  actor-only R1 as `AUTHORIZED / DISPATCHED — RESULT PENDING`.
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. The original documentation candidate
  `61eaa3fe1bdeb7aef3cfb7c10c4d8964caf2ed0e` (committed `2026-08-30 23:31:17 +0300`) carried
  the record; GPT requested review fixes, and the correction landed as the DIRECT CHILD
  COMMIT `88352b2f…`, which is the APPROVED head — the held-out band, the manifest caller,
  the R1 scale state and the Task-5B label.
  **THE RETARGETING CHANGED THE BASE, NEVER THE CANDIDATE.** The branch was originally
  stacked on the PR-#43 branch and was RETARGETED to `main` once PR #43 was merged; because
  changing a PR's base invalidates a base-relative verdict even when the head SHA does not
  move, the unchanged approved head was **EXACT-BASE RE-REVIEWED against `main` before
  merging**, and the head remained `88352b2f…` throughout.
  The candidate was merged with a normal MERGE COMMIT and preserved as its SECOND PARENT
  (ordered parents: `b3c2e01f130afe854b09384cd6e1e196de714795`, then
  `88352b2fc03174e8095d3c7e8a1ef58b60e58e0b`); candidate and integration share the IDENTICAL
  tree `f57c472c73e236867064e9fc26426482f538152d` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #44**, and none may be
  inferred from it: the R1 long run remains `AUTHORIZED / DISPATCHED — RESULT PENDING`, with
  **no reward, convergence, attrition, benchmark or validity result stated or inferable**.
  Task 5A / Task 5B remain ENGINEERING VALIDATION under their binding label. This entry
  records the LOCK; §8 owns the phase state.

- **GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN — EXECUTED / INDEPENDENTLY REVIEWED /
  `APPROVE — VALID MEASUREMENT`. THE FIRST SCIENTIFICALLY VALID GENERALIZED-V1 MEASUREMENT,
  AND ITS PRIMARY FD FINDING IS NEGATIVE.** The measurement is attributable to exact code
  SHA `4af6c5aa5dd28072692bfda63282964b55010aae` — the approved PR-#43 candidate, a durable
  MEASUREMENT identity and **never a claim about live `main`**. Per §7's hash convention this
  entry is keyed to the MEASURED CODE SHA; the documentation commit that creates it, and the
  merge that integrates it, cannot name their own SHAs and are deliberately not invented
  here. **No tracked file changed for the measurement** — it is a run of already-reviewed
  merged code, not a candidate.
  **THE RUN SHAPE IS THE FROZEN PLAN, UNCHANGED.** `training_mode = actor_only`, a FIXED
  budget of 375 iterations × 8 SUCCESSFUL episodes per iteration, `episode_design =
  generalized_v1`, `fuel_damage_mode = seeded_variable`, **NO early stopping**, **NO CTDE
  arm**, `worlds_per_cell = 3`, evaluation and checkpointing every 25 iterations.
  **VALIDITY VERDICT — `APPROVE — VALID MEASUREMENT`, judged VALIDITY BEFORE PERFORMANCE.**
  - **375 / 375 scheduled iterations completed, and 375 / 375 PPO updates**, so the fixed
    budget was consumed in full and no iteration was lost.
  - **3000 successful training episodes from 3045 attempted** — exactly the quota the
    `successful_quota_with_deterministic_replacement_v1` policy requires, with **45 ordinary
    accounted `setup` failures, every one of them DETERMINISTICALLY REPLACED** by the next
    run-wide attempt ordinal. `3000 + 45 = 3045` reconciles by construction, each failure was
    recorded ONCE in `episode_failures.jsonl`, no seed was retried, no seed was substituted
    and no band shifted. The attempt count sits well inside the MAXIMUM POSSIBLE band of
    `375 × 12 = 4500` (§5).
  - **ZERO integrity aborts.** No `MeasurementIntegrityError`, no `EpisodeRosterError`, no
    `TrainingQuotaError`, no `FuelDamageIntegrityError`, no `BenchmarkIdentityError`, no
    aborting `ReferenceIntegrityError` and no `_VisualArtifactError` — so no episode was
    removed from a scientific population by an instrument fault, which is precisely the
    failure that made the first long baseline INCONCLUSIVE (§7, §8).
  - **16 evaluation rounds** — the initial `pre_update` plus one every 25 iterations across
    375 — with **864 / 864 benchmark members successful** and **18 / 18 COMPLETE matched
    clean/mild/severe groups in EVERY round**. `16 × 18 × 3 = 864` reconciles, so the
    comparator was measured in full, no member failed, no group was incomplete and no delta
    was taken over a repaired group.
  - **`accounting_reconciled = true`.**
  **THE FROZEN COMPARATOR, BY CONTENT-ADDRESSED IDENTITY.** The R1 benchmark manifest was
  built by the deterministic preflight BEFORE training and evaluated unchanged in every round:
  `manifest_id = 0e15f007ef176bf977f8b93bb91289f48c16f25ee9eee282ffd1a89477f6fc0d`;
  manifest file `SHA-256 = 76768cfd311686a51fc79b82e4bb5142dd4931fa5bb7f151a32b11106195e11d`;
  ordered world-seed digest
  `seed_list_sha256 = c417683520bd89f4074d53652df719e6cf556808c29f0335b7fc728ce153fbb1`;
  preflight report
  `SHA-256 = f2041b97bc34c8a1750daa2135468b6ed5d2329d9089bd377517a5ebda43f903`.
  **`manifest_id` IS THE HASH OF THE CANONICAL PAYLOAD AND IS NOT THE HASH OF THE FILE**
  (§5) — the two values above are therefore DIFFERENT quantities and neither substitutes for
  the other. **NO BENCHMARK MANIFEST IS COMMITTED OR TRACKED IN THE REPOSITORY**, and this
  entry records the comparator's IDENTITY rather than adding its bytes to the repository.
  **NO REVIEW-BUNDLE ZIP HASH IS RECORDED HERE.** It is deliberately omitted rather than
  quoted: no preserved review-bundle artifact exists in this workspace to derive it from,
  and a hash may never be expanded from chat or memory. Its absence is a statement about
  THIS record, not about the review, which was performed on evidence the reviewer held.
  **THE PRIMARY BEHAVIOURAL RESULT IS NEGATIVE, AND IT IS A RESULT.** The run **did NOT
  learn severity-conditioned mild-vs-severe behaviour.** The policy moved GLOBALLY from
  `SELF_PRESERVATION_ABORT` toward `PLAN_COMPLIANCE` across checkpoints, and it treated
  MATCHED mild and severe worlds **almost identically** — a global shift, not a
  severity-conditioned one. **THIS IS A VALID NEGATIVE RESULT, NOT A VALIDITY DEFECT**, and
  it is **not** a technical failure, **not** evidence that training or PPO failed, **not**
  evidence that the actor ignores fuel entirely, and **not** grounds to re-tune, re-seed,
  repair, resume, extend or re-run. **NO RERUN, REPAIR, RESUME, EXTENSION OR RETUNING IS
  AUTHORIZED**, and each would be a separate research decision requiring its own explicit
  authorization.
  **SCOPE OF THE CLAIM, STATED AS NARROWLY AS THE EVIDENCE ALLOWS.** This is **ONE R1
  MEASUREMENT**. It is **NOT** a five-run population result, and it is **NOT** an
  actor-only-vs-CTDE comparison — **no CTDE arm was run, and no CTDE benefit or deficit is
  established, supported or pre-claimed by it.** The unchanged interpretation rules apply: a
  mean is never read without its denominator, an all-failed batch is `null` and never `0.0`,
  within-world claims come only from COMPLETE matched groups, and FD-wake rates are reported
  over FD WAKES.
  **THE DIAGNOSTIC REPLAY — ENGINEERING / ANALYSIS EVIDENCE, NOT A SECOND MEASUREMENT.** A
  bounded offline replay was performed against R1's own checkpoints, manifest and measured
  code SHA to ask WHY the response was flat. It is labelled engineering/analysis evidence
  because that is its designated purpose: it schedules no population, defines no comparator
  and produces no scientific verdict, and **no reward, learning or performance claim may be
  drawn from it.** What it established:
  - **REPLAY EQUIVALENT TO R1** — **108 / 108 action matches**, with the event ticks and ego
    ids matching as well, so the replay reproduces the run it analyses rather than describing
    a neighbouring one;
  - the ego's own **`fuel_norm` differed MATERIALLY in ALL 54 matched pairs**, so the one
    input the decision is supposed to be read off really did change;
  - **`reachable_by_ego` FLIPPED in ALL 54 pairs**, so a second, structural input changed too;
  - and the **selected meta-action changed in 0 / 54 pairs**;
  - **mean absolute matched delta in aggregate P(ABORT) = 0.0001177037203753436** — a
    difference in probability MASS that is numerically negligible, and which is the aggregate
    column mass, **never** the probability of the selected action (§5);
  - **joint-cell vs aggregate-meta-action argmax disagreement: 54 / 108**, so the two views of
    the `k × 3` surface disagreed on half the decisions and must not be read as one quantity;
  - **mean task-distance clipping 98.15 %** — almost the entire `dist_to_ego_norm` column
    saturated at the fixed normalizer, which is a property of the NORMALIZER rather than of
    the policy;
  - **normalized joint entropy remained HIGH**, so the distribution did not collapse.
  **ACTION ALIASING AND WEAK ROUTE-RELATIVE OBSERVATION CONTEXT ARE SUSPECTS, NOT CAUSALLY
  PROVEN EXPLANATIONS.** Nothing in the replay establishes a cause; it narrows where to look.
  Diagnostic bundle `SHA-256 =
  812ff43322e134e9a7ca31720007393ff1220ba50c35955b2a724b30d4d5d792`.
  **PRESERVATION.** The R1 run tree and the diagnostic bundle are preserved and must not be
  modified, moved, copied, repackaged, deleted or regenerated, and neither may any earlier
  preserved run. The approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1 (`bf1e045f`)
  measurements are untouched by this entry and remain measurements of the `fixed_cell_v1`
  bundle. **§8 owns the phase state.**

- `81a148f8` — **GENERALIZED-V1 DURABLE PER-WAKE FD POLICY DIAGNOSTICS (MEASUREMENT
  HARDENING) — CLOSED / APPROVED / MERGED.** FINAL approved candidate SHA
  `81a148f80317499d8897db44bd713976962db832`, integrated by merge commit
  `28eb8dad2643fc79d516b47ec95119a395e76257` (PR #52), from base
  `44530abb1cc3f99d01ac867c6621047ac9343661` — the `main` head produced by the final
  early-stopping handoff-stabilization merge (PR #51). The candidate was merged with a normal
  MERGE COMMIT and preserved as its SECOND PARENT (ordered parents:
  `44530abb1cc3f99d01ac867c6621047ac9343661`, then
  `81a148f80317499d8897db44bd713976962db832`); candidate and integration share the IDENTICAL
  tree `86c3b04d104d38c6d6fc5c1e2bdda3bb5c1ab9b7` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. Grade A under `GPT_GITHUB`. The technical contract is in §5 (the
  GENERALIZED-V1 per-wake FD policy diagnostics block) and the routing in §6; this entry
  records the LOCK, not the mechanism.
  **APPEND-ONLY REVIEW CHAIN — FOUR COMMITS ON ONE BRANCH AND ONE PR**, never amend, rebase,
  squash, force-push or history rewrite. The original implementation candidate
  `b51515c1c78faa3354bfca71293897b910873c64` carried the layer, and three review-fix commits
  landed as DIRECT CHILDREN on the same branch — `039a3b6d99136eb47c32eee3af535dc4c4d9d872`,
  then `e1adb8e9078ec2a39bce880d01113d7e8ef6cfeb` (semantic final-eval selection,
  figure-consistent optional-plot declaration, and the exact top-two margin), then the
  APPROVED head `81a148f8…` (the FD digest's final round requiring the COMPLETE identity).
  **CUMULATIVE REVIEWED SCOPE: EXACTLY SEVEN FILES**, verified as the complete
  `44530abb…...28eb8dad…` comparison — `src/match_aou/rl/action/graph_action.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_ctde.py`,
  `tests/test_graph_fuel_damage.py`, `tests/test_graph_train.py` and
  `tests/test_graph_wake_diagnostics.py` (NEW). **No config, preset or benchmark manifest was
  added or changed** — `configs/graph_train/final_cell_probe.json` remains the ONLY repository
  preset, is untouched and is still `fixed_cell_v1` — and **no documentation file was part of
  the code integration**, which is what this documentation task closes. No vendored BLADE,
  solver, `graph_reward`, `graph_generalized`, `graph_benchmark_preflight`,
  `graph_episode_setup`, `graph_fuel_damage`, `graph_ppo`, `graph_encoder`, `graph_effect`,
  `graph_trigger`, executor, generator or rollout file was touched.
  **HISTORICAL CC-REPORTED ENGINEERING EVIDENCE ONLY.** As reported at review time: the final
  solver-free suite **602 passed, 6 skipped**, and the focused wake-diagnostics suite
  **57 passed**; and the `graph_tick_loop` BONMIN selftest was **NOT run in the final fix
  chain**. **This DOCUMENTATION task ran no test suite, no solver, no BLADE episode and no
  smoke, and makes no pass/fail claim of its own** — it confirmed only, read-only, that
  `tests/test_graph_wake_diagnostics.py` COLLECTS 57 tests at the integrated tree, which
  corroborates that count without asserting a result.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #52**, and none may be inferred
  from it: **PR #52 produced no scientific measurement and did not modify the R1 run, its
  artifacts or its verdict.** R1 was measured at code SHA `4af6c5aa…`, which PREDATES this
  layer, so R1's artifacts are episode-outcome schema v2 and carry no `wake_decisions`; the
  layer's benefit is to FUTURE runs. The approved Phase-A (`737b4bf`) and
  FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) measurements are untouched. This entry certifies the
  IMPLEMENTATION; §8 owns the phase state.

## 3. Phase records

- **PHASE A IS CLOSED. A SCIENTIFICALLY VALID LONG-BASELINE MEASUREMENT OF THE FUEL-DAMAGE
  CELL EXISTS (measured code SHA `737b4bf`, §7). THE ADDITIONAL ACTOR-ONLY
  FD-VARIABLE-SEVERITY-v1 BASELINE IS NOW ALSO EXECUTED, INDEPENDENTLY REVIEWED AND
  `APPROVE — VALID MEASUREMENT` (measured code SHA `bf1e045f`, §7) — WITH A NEGATIVE
  PRIMARY FINDING. AND PHASE-B CTDE IS NOW IMPLEMENTED, REVIEWED AND MERGED
  (`a6f3aa9` / `8390d85`, PR #30, §5 and §7) — SO THE NEXT SCIENTIFIC TASK IS THE FIRST
  CONTROLLED ACTOR-ONLY vs CTDE COMPARISON ON THE LOCKED ORIGINAL PHASE-A CELL, WHICH HAS
  NOT BEEN RUN AND FOR WHICH NO BENEFIT IS CLAIMED.** The variable-severity measurement ran
  on an immutable DETACHED snapshot while CTDE design and implementation proceeded beside
  it in a separate writable task branch. **The earlier serial claim — that CTDE may begin
  ONLY AFTER that measurement — was SUPERSEDED on 2026-08-22, and the CTDE INTEGRATION gate
  is now SATISFIED AND CLOSED ON BOTH HALVES**: the measurement-validity half by that
  `APPROVE — VALID MEASUREMENT` verdict, and the reference half by
  **`pre-ctde-actor-only = d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the FIRST parent of
  the CTDE integration, which must not move (`phase-a-baseline` remains the SEPARATE
  original Phase-A reference and was never repurposed). See the research-ordering bullet
  below for the historical arrangement, and the CTDE bullet below it for the live state.
  Difficulty selection is CLOSED
  (below) and FD-BASELINE-v1 is merged and locked (`a8669f4`, §7), so the open question was
  never *what* to build but *how the built cell behaves* — and for THIS cell that question is
  now ANSWERED by the approved rerun
  `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, whose full record,
  denominators, explicit NON-CLAIMS and evidence hashes are in §7. **FOUR runs of the
  LEGACY FD-BASELINE-v1 cell exist. The first three are NOT valid measurements and survive
  as HISTORY ONLY:**
  - **First short probe — `training_output_20260815_173029`**, from clean `main` at
    `238062d7d284334432d9c39d7543fb0bbf39ea7c`. It established HARNESS AND ACCOUNTING
    OPERABILITY ONLY and exposed three research-validity defects (next bullet), so **its
    reward numbers are NOT scientific evidence about the fuel-damage cell** and remain
    historical evidence about the PRE-CORRECTION behaviour.
  - **Corrected short-probe rerun — `training_output_20260816_162130`**, from a clean
    checkout at exact code SHA `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, one invocation of
    the reviewed preset through `--config`, native exit code 0, `cli_overrides = []`. It was
    originally reviewed as `VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS`; **that verdict
    is SUPERSEDED** by `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE
    SCIENTIFIC DENOMINATOR` (§7). Its own ledger accounts clean train seed 4 as a `setup`
    `EpisodeRosterError`, and the approved roster-integrity correction (`36365f2`)
    establishes that such a fault is a measurement/data-integrity failure that must ABORT —
    not an episode outcome that may shrink a denominator. **Its reward and performance
    numbers are therefore no longer scientific evidence**, and the claim that it PASSED or
    permanently released the long-baseline validity gate is WITHDRAWN. What it DOES still
    establish is preserved: its run identity, provenance, mechanical accounting, artifact
    completeness, and the OPERATIONAL WITNESSING of all three defect corrections in real
    playback.
  - **First long baseline — `training_output_long_baseline_100x8_seed0`**, from exact code
    SHA `c30b6982ba605d60976cc303256da4b5528b0e63`, one invocation with `cli_overrides = []`
    and native exit code 0. **EXECUTED, independently reviewed, engineering `REQUEST
    FIXES`, scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED`** (§7 owns the full
    record, contract, denominators, failure breakdown and evidence hashes). 1,136 scheduled
    attempts, 860 successful, 276 failed, `accounting_reconciled = true` and 100 PPO
    updates — and **143 training attempts were nevertheless destroyed by the roster defect
    across 83 iterations while the run reported itself healthy**, with 11 `complete`
    manifests claiming a five-target world their own authoritative executed-t0 scenarios
    contradicted. **Do not report its reward, paired deltas, survival, fuel-damage yield or
    PPO performance as scientific evidence**; they are raw historical outputs only. It is
    preserved and must not be modified, moved, repackaged, deleted or regenerated.
  - **Phase-A long baseline (RERUN) — `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`**,
    from a clean checkout at exact code SHA `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`, ONE
    invocation of the preserved measurement contract through `--config`, native exit code 0,
    `cli_overrides = []`. **EXECUTED, independently reviewed, `APPROVE — VALID
    MEASUREMENT`.** **THIS IS THE AUTHORITATIVE MEASUREMENT OF THE CELL** and the only one
    whose reward, paired-delta, survival, event-yield and PPO numbers are scientific
    evidence. §7 owns the full record; the headline is 1,136 scheduled / 993 successful /
    143 accounted episode failures with `accounting_reconciled = true` and ZERO
    infrastructure or data-integrity faults, 100/100 productive PPO updates, a matched
    paired delta of `−0.375000 → −0.071429` over a structural 7/8 pairs, and evaluation
    deaths `7 → 0`.
  **TWO FURTHER runs exist on the VARIABLE-SEVERITY design, and they measure a SEPARATE
  cell — never a rerun, replacement or extension of the four above:**
  - **Invalid variable-severity precursor —
    `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640\training_output_fd_variable_severity_v1_50x8_seed0`**,
    at exact code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`. **`INCONCLUSIVE/BLOCKED —
    INVALID MEASUREMENT`.** A Windows `MAX_PATH` playback-export failure (267-character
    recording path) produced 70 `run`-stage `FileNotFoundError`s that were ALL
    `post_update` SEVERE members — 10 rounds × the 7 feasible held-out seeds — so the
    entire post-training severe arm, the arm the experiment exists to measure, was
    systematically removed. **That is an infrastructure failure, NOT a negative scientific
    result.** Preserved as engineering history only; §7 owns its hashes.
  - **FD-VARIABLE-SEVERITY-v1 actor-only baseline (REPLACEMENT) — run root
    `C:\Users\Itama\f7r2`**, same exact code SHA
    `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, ONE invocation of a contract differing from
    the precursor's in `output_dir` ALONE, native exit code 0, `cli_overrides = []`.
    **EXECUTED, independently reviewed, `APPROVE — VALID MEASUREMENT`.** **THIS IS THE
    AUTHORITATIVE MEASUREMENT OF THE VARIABLE-SEVERITY CELL.** §7 owns the full record; the
    headline is **664 scheduled / 586 successful / 78 accounted `setup` episode failures**
    with `accounting_reconciled = true` and ZERO infrastructure or data-integrity faults,
    **7/8 complete matched triads in all 11 rounds**, 50/50 productive PPO updates — and a
    **NEGATIVE primary finding: NO severity-conditioned FD-wake meta-action separation**,
    with MILD and SEVERE both at `PLAN_COMPLIANCE 7/7` at `pre_update` and at the final
    `post_update`, and identical `63/70` vs `7/70` distributions across all ten
    `post_update` rounds. Physical outcomes nevertheless DIVERGE sharply, so the severity
    factor itself is real. Its numbers are evidence about the VARIABLE-SEVERITY cell only,
    never about the Phase-A legacy cell.
  **What this establishes and what it does not.** The roster/world-truth defect was closed in
  CODE (`36365f2`, integrated `f37ea1c`, PR #24 — §5, §7), the authorized rerun was then
  executed ONCE on the SAME scientific contract into a NEW output directory, and it PASSED
  the validity gate. **Phase A is therefore CLOSED**, and **the long baseline is NOT to be
  re-run, resumed, repaired, extended or re-tuned** — a valid measurement exists, and
  re-running it would not make it more valid. The approved result establishes end-to-end
  learnability and meaningful ego-local runtime adaptation in the LOCKED Phase-A reference
  cell. It establishes **NO global optimality, NO monotonic convergence, NO generalization
  beyond this fixed cell and this held-out seed set, and NO benefit from centralized
  training** (§7 states these non-claims in full, and they must be carried forward verbatim
  in meaning). The interpretation rules survive unchanged: a held-out mean is never read
  without its denominator; an all-failed batch reports `null`, never `0.0`; an empty
  successful-pair population is `null` too; the held-out per-condition means are each over
  their own successful subset, so the within-seed claim is the matched-pair delta over
  COMPLETE pairs alone; and the `static_t0_v1` reward path this baseline was measured on
  remains FROZEN unless a separately reviewed design requires an explicit reward-contract
  change. *(One such reviewed change has since landed and is scoped so that it does NOT
  touch this path: GENERALIZED-V1 Task 3 (`24a8b1e`, §5, §7) added an OPT-IN
  `event_conditioned_continuation_v1` reference beside the static one. The static formula
  is byte-unchanged, and Task 4 (`db79013`, §5, §7) then exposed that reference to the
  harnesses through the `episode_design` selector — `fixed_cell_v1` is the DEFAULT and
  resolves `static_t0_v1`, so reaching the opt-in reference requires explicitly naming
  `generalized_v1`, and this measurement is unaffected. `p(destroy) < 1` remains separately
  deferred.)* The invalid old long run may be
  compared against **only as ENGINEERING evidence** — never as a scientific baseline.
  **The B2 exact-cardinality and fuel-window failures BOTH long baselines recorded — 101 and
  32 in the invalid first run, 101 and 42 in the approved rerun — are NOT corrected by
  `36365f2`, and they are NOT defects.** They remain EXPECTED SCIENTIFIC
  OUTCOMES under the current contract — `skip_and_account_v1` attempts each seed once,
  records it once and reports the smaller successful population next to its denominator (§5,
  and the exact-cardinality bullet below) — and they must not be relaxed, retried, retuned
  or reclassified. Only the ROSTER fault changed category, because only it was an instrument
  defect.
  **The deferred over-safety observation is a HYPOTHESIS, not a defect and not a semantic
  change.** The corrected rerun's playback shows a B-2 at 50.07 km from a second hidden
  target — against the 50 km `DETECTION_KM` threshold — on the sampled frame before it
  enters RTB, and the preserved artifact does not persist that wake's selected meta-action,
  so the attribution is PLAUSIBLE BUT NOT PROVEN (§7). It is recorded as a future research
  hypothesis about policy calibration, relevant to a later variable-FD-severity experiment.
  **It opens no defect, changes no reward, retunes no policy, and it is not what
  invalidated either measurement** — the roster/data-integrity fault is. It neither blocked
  nor shaped the Phase-A long baseline, and it remains a deferred hypothesis for a later
  variable-FD-severity experiment.
  **The harness every one of these runs used is MERGED and LOCKED** (`61e539e`, §7): driven
  through `--config` — the SHORT PROBES by the repository preset
  `configs/graph_train/final_cell_probe.json`, the TWO LONG BASELINES by their own
  measurement contract, which is deliberately NOT a repository preset — and writing
  `run_config.json` (with `provenance` and a structured `config_source`), the three jsonl
  records, `run_summary.json`, `scenarios/`, `checkpoints/` and the three figures under
  `plots/`. Two reading rules survive unchanged: the PRESET schedules TWO ITERATIONS, which
  does not guarantee two PRODUCTIVE PPO updates (`updates_completed` may be 0, 1 or 2 —
  the corrected rerun measured 2, but that yield is always a MEASUREMENT, never an
  assumption); and the held-out per-condition means are each over their own successful
  subset, so the within-seed claim is the matched-pair delta over COMPLETE pairs alone
  (§5). What counts as a VALID measurement — as opposed to a favourable one — is stated in
  the handoff: a run that produces no reward improvement, or no productive update, is a
  valid NEGATIVE observation, not a technical failure. A run whose DENOMINATOR was corrupted
  by an instrument defect is the opposite case — not a negative result but no result, which
  is exactly why `MeasurementIntegrityError` now aborts instead of being accounted (§5).
  All four runs used `--visual-artifacts` (§5, `24d1835`; the repository preset enables
  it, and both long-baseline contracts set it explicitly). It preserves each attempt's
  known-only scenario, executed t=0 scenario and BLADE playback for inspection, and it is an
  observation surface only: enabling it neither authorizes a run nor changes anything a run
  measures, and artifact completeness is reported ALONGSIDE the scientific denominators,
  never in place of one. **It is also what made the roster defect provable** — the preserved
  authoritative executed-t0 scenarios are what showed the full six-target world behind every
  under-counted roster.
  *Historical, and about the EASY PRE-FD CELL only:* the clean-code probe at
  `a3f0838616990987bcb8a51665fa75d84edf5952` measured pre-update headroom
  (`-0.4999997395829586`, 4/4), train yield 7/8 with one accounted seed-2 `setup` failure,
  24 transitions, two productive PPO updates and a final held-out numerical zero
  (`5.000007394910353e-7`, 4/4). That cell had no difficulty factor; **those numbers are
  not evidence about the fuel-damage cell** and must not be reused as its baseline.
- **THE AUTHORIZED MEASUREMENT CONTRACT — the bounded actor-only FD-VARIABLE-SEVERITY-v1
  baseline. EXECUTED ON THE PINNED IMMUTABLE SNAPSHOT; INDEPENDENTLY REVIEWED
  `APPROVE — VALID MEASUREMENT`; §7 OWNS THE RESULT.** The measurement was LOCKED to exact
  SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
  `dd881478b8e2e521054d09bc865437f1308be1a2`, in a DETACHED, clean snapshot worktree that
  was READ-ONLY with respect to the shared repository. **This bullet remains the run SHAPE
  and nothing more** — the executed run's identity, denominators, evidence hashes and its
  NEGATIVE primary finding live in §7, and no number may be quoted from here. Because the
  snapshot was pinned and detached, later `main` work — Phase-B CTDE included — lies
  outside the measured tree and can neither be attributed to that run nor contaminate it.
  **The measurement is NOT to be re-run, resumed, repaired, extended or re-tuned:** a valid
  measurement exists, and a NEGATIVE finding is a result, not a reason to run it again. The
  approved shape, which the executed run followed exactly:
  - **50 scheduled training iterations × 8 scheduled training attempts = 400 scheduled
    training attempts**, `base_seed = 0`;
  - **evaluation every 5 iterations, INCLUDING the initial `pre_update` round ⇒ 11
    evaluation rounds**;
  - **8 fixed held-out seeds in the EXISTING eval band**, each evaluated as a matched
    **clean / mild / severe TRIAD** ⇒ **11 × 8 × 3 = 264 scheduled evaluation attempts**;
  - **664 scheduled attempts in total; NO early stopping.**
  Training runs `fuel_damage_mode = seeded_variable` at the approved
  **0.50 clean / 0.25 mild / 0.25 severe** distribution (§5). Everything else is the
  LOCKED cell: 3 agents, 3 known + 3 hidden, 200 km / 100 km geometry,
  `DETECTION_KM = 50`, `include_sams = false`, `probability = 1`, frozen solver and BLADE,
  unchanged `graph_reward` formula with `aircraft_penalty_coeff = 2.25`, unchanged PPO.
  The run task chose a FRESH, NON-OVERWRITING output directory and captured its own
  provenance; §7 records which. The interpretation rules carry over unchanged and are not
  optional: the PRIMARY behavioural evidence is the severity-conditioned FD-WAKE
  meta-action response with its own FD-wake denominators; a mean is never read without its
  denominator; the only within-seed claims are the three deltas over COMPLETE triads; an
  empty population is `null`, never `0.0`; and a run that shows no reward improvement, no
  productive update, or **no severity-conditioned behavioural difference — which is what
  this one measured** — is a valid NEGATIVE observation, not a technical failure. **"Mild
  must choose `PLAN_COMPLIANCE`" is NOT a correctness criterion** (§5). One further rule
  the executed run makes concrete: the ten `post_update` rounds REUSE the same seven
  feasible held-out seeds, so 70 observations per severity are a TRAJECTORY across
  checkpoints and **not 70 independent worlds** — the clean statistical unit for the final
  policy is the final round's 7 complete triads (§7).
- **Post-B3 headroom exists — CLOSED first by the `dd14ab4` reference and then measured
  by the `a3f0838` probe.** The default cell emits `n_known + n_hidden` = 3 + 3 = 6
  targets. The B3 seed-0 reference completed `ended=done` with 4 organic wakes,
  `u_achieved=320`, `U_oracle=479.99968` and reward `-0.3333`. The later fixed-band
  probe measured the untrained deterministic policy at
  `pre_update=-0.4999997395829586` over 4/4 and the same band after two PPO updates at
  `post_update=5.000007394910353e-7` over 4/4. Thus the cell had real headroom and the
  loop could close it in a short run. Neither result is a baseline: one is a single
  rollout and the other is a 2×4 diagnostic probe on the easy reference cell.
  **Both predate FD-BASELINE-v1** (`a8669f4`): the target counts are unchanged, but that
  cell carried NO difficulty factor and no death penalty, so neither number describes the
  fuel-damage cell's headroom. That headroom is now measured by the approved Phase-A long
  baseline (`737b4bf`, §7), which is the only valid source for it.
  *Historical, pre-B3 only:* the cell emitted no hidden targets and measured 0 wakes at
  `reward=+0.0000` because nothing existed to discover; that result is invalid as
  learning evidence. The authoritative target-count fields for future runs are the
  PR-#7 unique-target aggregates, not the probe's old confirmation-count aliases.

## 4. First short-probe defect observations

From the former handoff §3d.

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
  rerun is recorded in §3e**; closing the three defects never constituted a measurement by
  itself.
- **The gate language written in this section is SUPERSEDED.** At the time it was written
  the long baseline was blocked on that rerun, and the rerun was then read as releasing it.
  Neither statement survives: the rerun's verdict is SUPERSEDED (§3e), a first long baseline
  was subsequently RUN and is `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED` (§3f), and the
  authorized rerun that followed the PR #24 correction is the one that PASSED the validity
  gate (§3h). Nothing in this section is outstanding work.

## 5. The R1 frozen plan

From the former handoff §3m.4. R1 executed this plan unchanged.

**THE FROZEN PLAN, recorded so the eventual artifacts can be checked against what was
authorized:**

| Knob | Value |
|---|---|
| `training_mode` | `actor_only` |
| iterations | 375 |
| SUCCESSFUL episodes per iteration | 8 |
| total successful training episodes | 3000 |
| `generalized_max_attempts_per_iteration` | 12 |
| training base seed | 740000 |
| MAXIMUM POSSIBLE training seed band | `[740000, 744500)` |
| `worlds_per_cell` | 3 |
| R1 benchmark base seed | 840000 |
| `max_candidates_per_cell` | 12 |
| evaluation | every 25 iterations |
| checkpoint | every 25 iterations |
| early stopping | none |
| solver timeout | none adopted |
| CTDE arm | **not run** |

The seed band is the **MAXIMUM POSSIBLE ATTEMPT BAND** — `375 × 12 = 4500` — because a
failed replacement attempt still spends a seed (`CLAUDE.md` §5). **No CTDE run exists, is
scheduled or is authorized**, and **no actor-only-vs-CTDE generalized result exists.**

## 6. The P1 arms

### 6.1 The aborted arm

From the former handoff §3o.4.

**ONE ATTEMPTED FULL P1 ARM WAS ABORTED DURING TRAINING BY `FuelDamageIntegrityError`.**

- **IT IS NOT A COMPLETED SCIENTIFIC MEASUREMENT.** It carries no verdict, it was never
  submitted for independent review as a measurement, and **no reward, learning, attrition,
  convergence or comparison number from it may be reported.**
- **IT MUST NOT BE RESUMED, REPAIRED, CONTINUED OR EXTENDED AND THEN SILENTLY TREATED AS
  ONE.** **RESUME IS NOT AUTHORIZED**, and checkpoint RESUME remains out of scope in any case
  (`graph_train` is still SAVE-only).
- **ITS ROOT CAUSE IS CLOSED** (§3o.3): the execution accumulated **two** skipped engine
  updates before the certified event, so its **PHYSICAL certificate state was correct while
  its OUTER TICK was late** — certified event tick 914, crossing observed at outer tick 916,
  position matching to ~7e-11 km and pre-damage fuel to ~6e-9 lbs. **THE INSTRUMENT PREMISE
  WAS WRONG, NOT THE WORLD** — so the abort was correct behaviour under the old contract,
  and it is not a finding about P1.
- **THE FIX IS INTEGRATED THROUGH PR #55** and changes LIVE integrity semantics only (§3o.2).
- **A FUTURE FRESH P1 FULL RUN IS A NEW MEASUREMENT UNDER THE REPAIRED INSTRUMENT**, with
  its own frozen contract, **an EXPLICITLY RESOLVED AND FROZEN P1-SPECIFIC BENCHMARK
  CONTRACT**, and its own independent review. **NAMING IT HERE IS NOT AUTHORIZATION TO
  EXECUTE IT**, and **this documentation record neither launches it nor schedules it.**
- **THE BENCHMARK DECISION IS DELIBERATELY NOT TAKEN IN THIS DOCUMENTATION LOCK.**
  **Benchmark / manifest identity MUST be resolved EXPLICITLY before any execution.**
  **Whether the already-existing P1-specific benchmark is REUSED, INDEPENDENTLY REVALIDATED,
  or DETERMINISTICALLY REBUILT is a SEPARATE pre-run orchestration / research-validity
  decision** — this record does **NOT** decide it, does **NOT** pre-authorize any of the
  three, and does **NOT** schedule one. **NO SILENT POPULATION REPLACEMENT OR REGENERATION IS
  ALLOWED**, and no wording here may be read as committing the next orchestration to building
  a new benchmark population. **This record neither rebuilt, inspected by execution,
  regenerated nor altered any benchmark or manifest**, and it records no benchmark hash or
  artifact claim of its own.

### 6.2 The fresh valid arm and the provenance of GENERALIZED-V2

From the former handoff §3p.2.

Recorded because it is the reason the design was taken, and **kept deliberately minimal**.

- **A FRESH deterministic-P1 full arm was subsequently COMPLETED and independently ACCEPTED
  as a VALID MEASUREMENT, at measured code SHA
  `ae1941035991df4719df212c4b5dd07db89aee4a`**, and **its primary policy result was
  NEGATIVE for MILD-vs-SEVERE deterministic action separation.** That SHA is a real
  repository state — the `main` head PR #57 was based on — at which the MATCH-AOU backend
  selector already existed with `legacy_minlp_v1` as the DEFAULT, so the P1 objective was
  EXPLICITLY SELECTED there and never inferred. **This SUPERSEDES, as CURRENT state only, the
  2026-09-06 statements that no approved P1 measurement exists and that a fresh P1 full-arm
  orchestration is the next scientific thread** — each remains accurate as the record it
  was. It is a **SEPARATE run** from the earlier attempted arm recorded in §3o.4, which was
  ABORTED by `FuelDamageIntegrityError`, is **NOT a completed measurement** and remains
  **`DO NOT RESUME`**. **This document records no identity, denominators, artifacts or
  metrics for the fresh arm, and none may be invented.**
- **The P1 solver made reference solving computationally negligible.**
- **R1 and the fresh P1 arm are DISTINCT repository / population measurements and are NOT a
  clean causal solver-quality comparison.** **R1 was measured at
  `4af6c5aa5dd28072692bfda63282964b55010aae`; the fresh valid deterministic-P1 arm was
  measured at `ae1941035991df4719df212c4b5dd07db89aee4a`.** The relevant experimental /
  backend difference includes the **deterministic `p1_milp_v1` allocation semantics versus
  R1's legacy objective** — R1's measured state PREDATES the backend seam entirely, so its
  legacy objective was the only one available there — and the two SHAs differ by everything
  merged between them besides. Because **allocation semantics FEED route-relative
  construction**, the frozen worlds / populations the two arms ran over are **not
  identical**. **THEREFORE THIS IS NOT A LITERAL ONE-CONFIG-FIELD CONTROLLED COMPARISON**:
  no solver equivalence and no one-config-field experimental equivalence is claimed, and
  **no causal reward-difference or solver-quality inference is authorized.** **The earlier
  ABORTED P1 arm is a DIFFERENT, INVALID, `DO NOT RESUME` attempt and must never be
  conflated with this measured SHA.**
- **A later READ-ONLY construction audit established, IN THE AUDITED CURRENT V1 DOMAIN
  ONLY, that hidden shortfall was structurally explained by ROUTE-COUNT CAPACITY
  (`no_route`) rather than by hidden-placement geometry.** That is the finding V2's
  route-relative bound answers, and it is **an observation over an audited domain, never a
  universal mathematical guarantee.** V2 answers it by drawing `H` against the realized
  `R` — **not** by restoring the legacy MATCH-AOU redundant-stacking incentive — and still
  makes **no promise that `H_realized == H_requested`.**
- **Engineering scaling showed P1 remained cheap beyond `A = 4`**, but **those larger cells
  were NOT thereby scientific GENERALIZED-V1 measurements.** Supported V2 training
  cardinality stops at **`A <= 6`**; **`A = 8` and `A = 10` are ENGINEERING SCALING EVIDENCE
  ONLY** and must never be described or implied as supported training cells.

## 7. GENERALIZED-V2 development R1 arms

Recorded in this restructure from the evidence refs of PR #61 and PR #62, both draft and
unreviewed on 2026-09-14. This section restates artifact contents; **it is not a validity,
behaviour or comparison verdict**, and the research decision that authorized the preflight and
these runs is not recorded in the repository.

- **Identities.** Both runs record measured code SHA
  `ae42cb01677f94868b2873008d87be677e31f0c8` with `dirty = false` on branch `main`. Each
  evidence commit is a single commit whose parent is that SHA and whose tree adds only
  `research_evidence/generalized_v2/actor_only_dev_r1/` (PR #61, head
  `1375a881637a9a32721a1630f598adc571422a47`) or `research_evidence/generalized_v2/ctde_dev_r1/`
  (PR #62, head `b2bbe7a6235c3b9255106826cfb268af7e73f72d`).
- **Shared frozen contract** (from both `run_config.json`): `episode_design = generalized_v2`,
  `match_aou_backend = p1_milp_v1`, `benchmark_profile = development`, `eval_seed_source =
  benchmark_manifest`, `manifest_id ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`,
  `n_iterations = 375`, `episodes_per_iteration = 8`,
  `generalized_max_attempts_per_iteration = 12`, `base_seed = 3000000`, `eval_every = 25`,
  `checkpoint_every = 25`, `early_stopping = false`, `fuel_damage_mode = seeded_variable`. The arms
  differ in `training_mode` (`actor_only` versus `ctde`).
- **Benchmark manifest.** Not committed. PR #62's `artifact_sha256.txt` records the external file
  `C:/Users/Itama/PycharmProjects/graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0/benchmark_manifest.json`,
  204 244 bytes, SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`, with
  the manifest id above. The file was not inspected by GPT or in this restructure, it is not part
  of either evidence package, and the code SHA that produced it is **unverified**: the directory
  name ends in `ae42cb0`, but no preflight report or run configuration for it is in the
  repository. *(Superseded on 2026-09-15 for the producer SHA and clean state only — see
  §8.11.)*
- **Accounting** (from both `run_summary.json`): `updates_completed = 375`; training 3008
  attempted / 3000 successful / 8 failed, all at stage `setup` with `FuelDamageError`; evaluation
  960 attempted / 960 successful / 0 failed over 16 rounds (consistent with the development
  profile's 20 groups × 3 members per round); `accounting_reconciled = true`;
  `episode_outcomes_recorded = 3960`.
- **Evidence integrity.** Each `episode_outcomes.jsonl` is committed as eight line-aligned shards
  whose concatenation reproduces the source SHA-256 (actor-only
  `b1bcc0647c45932e1ad6c939d0c740100291dc60a6a83371efb6485d8ebb9da3`, 73 894 321 bytes; CTDE
  `fbb18848eef09f187f9335eaed1aa97d686b1ae4494484e06d333afc27d6327b`, 73 969 142 bytes). Final
  checkpoints are recorded by hash only (actor-only
  `ae4111e089f4e5f23910997b0c10c420dc71788067a1ccfb56fac42fd10ae3f1`; CTDE
  `6555e0c7bf0546ece0efee4ec0e12fcea99591a6a49fd177856d7bf85fb10cc6`).
- **Verdict provenance.** CTDE: PR #62's body and `artifact_sha256.txt` state a prior GPT verdict
  `APPROVE — VALID DEVELOPMENT MEASUREMENT`; GitHub holds no review or comment record for either
  PR. Actor-only: no verdict appears in PR #61 or its `artifact_sha256.txt`.
- **Launch record.** PR #62 preserves a failed first CTDE launch as a pre-execution launch
  failure — a relative-import `ImportError`, exit code 1, before the run directory existed — and
  not a scientific run.
- **Known artifact defect.** Both `run_summary.json` files carry the wrong
  `generalized.cardinality_sampler` label
  ([artifacts and metrics §6.1](../contracts/artifacts_metrics.md#61-known-summary-label-defect-run_summaryjsongeneralizedcardinality_sampler)).

## 8. GENERALIZED-V2 development closure

Recorded on 2026-09-15. It extends §7, which stays as the record it was on 2026-09-14. The
decisions taken on this record are in
[`decisions.md` §1](decisions.md#1-decision-log) (2026-09-15 rows).

### 8.1 Scope and epistemic status

Several distinct things are recorded here. Each has its own status, and none upgrades another:

| Layer | What it is | Status |
|---|---|---|
| technical completion and accounting | the five runs finished and reconciled their own counts | established from each run's own artifacts (§8.3) |
| evidence preservation | PR #61, #62 and #64 carry copies of run files | PR #64's preservation candidate `90516d51…` was approved, as stated in the user-transferred packet of 2026-09-15; GitHub holds no review or comment record on any of the four PRs |
| analysis package | PR #65 extracts per-wake rows and matched pairs from recorded artifacts | extraction integrity `APPROVE` at `d565174e…` (same attribution); a **temporary** package, not a permanent evidence archive |
| scientific interpretation | what the combined development data are consistent with | a **development** interpretation, conditioned on the recorded frozen manifest (§8.10) |
| confirmatory evidence | — | **none exists**. The confirmatory profile was not used |

**The three overnight arms are development diagnostic runs**, executed by one sequential
LOCAL `nlp_env` sweep between 2026-09-13 22:20 UTC and 2026-09-14 03:27 UTC. They are not
confirmatory measurements. **Repeated evaluation rounds re-measure the same frozen development
worlds.** Cross-round totals below describe repeated measures, never independent worlds
([`experiments.md` §4.3](../workflows/experiments.md#43-interpretation-rules)).

Every number in §8.3–§8.9 was checked on 2026-09-15 against the relevant preserved evidence
files at the evidence heads named in §1. For the two R1 arms (PR #61, PR #62), those heads provide
the run configuration, summary, evaluation, training and episode evidence used here. For the three
diagnostic arms, PR #64 additionally provides each arm's `accounting_check.json`. PR #65 provides
`immediate_fd_wakes.jsonl`, `matched_mild_severe_pairs.jsonl` and `extraction_summary.json`. No
scientific execution was needed or performed for that check.

### 8.2 The five runs

All five record measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8` with
`provenance.git.dirty = false`. Evidence-commit SHAs are ledger locations, not measurement
identities ([artifacts and metrics §6.2](../contracts/artifacts_metrics.md#62-measured-code-sha-versus-evidence-commit-identity)).

| Variant | `training_mode` | Evidence ref at review | Local original |
|---|---|---|---|
| actor-only R1 | `actor_only` | PR #61 @ `1375a881637a9a32721a1630f598adc571422a47` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_actor_only_dev_r1_seed3000000_ae42cb0` |
| CTDE R1 | `ctde` | PR #62 @ `b2bbe7a6235c3b9255106826cfb268af7e73f72d` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_r1_seed3000000_ae42cb0` |
| `smallbatch` | `ctde` | PR #64 @ `90516d51beeddacded2b89a321d14291e411f2b0` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_smallbatch_seed3000000_ae42cb0` |
| `largebatch` | `ctde` | PR #64 (same) | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_largebatch_seed3000000_ae42cb0` |
| `fd80` | `ctde` | PR #64 (same) | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_fd80_seed3000000_ae42cb0` |

PR #64 does not commit the three arms' `episode_outcomes.jsonl`, checkpoints, plots or
`scenarios/`. It records their hashes in its `artifact_sha256.txt`.

### 8.3 Shared configuration and accounting

- **Controlled R1 pair.** Actor-only R1 and CTDE R1 are GENERALIZED-V2 development-profile runs
  with the same recorded frozen manifest, seed and training budget, differing in
  `training_mode` (§7).
- **Common to all five runs:** `episode_design = generalized_v2`, `match_aou_backend =
  p1_milp_v1`, `benchmark_profile = development`, `fuel_damage_mode = seeded_variable`,
  `fuel_damage_mild_probability = 0.5`, `base_seed = 3000000`, `early_stopping = false`,
  `held_out_verified = true`, `manifest_id
  ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`.
- **Identical accounting in all five `run_summary.json`:** training 3008 attempted / 3000
  successful / 8 failed, all `FuelDamageError` at `setup`; evaluation 960 / 960 / 0 over 16
  rounds (1 `pre_update` + 15 `post_update`); `accounting_reconciled = true`; early stopping not
  triggered. Each diagnostic arm's own read-only `accounting_check.json` reports `PASS` with an
  empty `hard` list.
- **What differs** (`run_config.json:/train_config`; transitions per update =
  `run_summary.json:total_transitions / updates_completed`):

| Variant | `n_iterations` | successful episodes / update | `generalized_max_attempts_per_iteration` | `fuel_damage_probability` | updates completed | transitions / update |
|---|---:|---:|---:|---:|---:|---:|
| actor-only R1 | 375 | 8 | 12 | 0.5 | 375 | ≈ 24.8 |
| CTDE R1 | 375 | 8 | 12 | 0.5 | 375 | ≈ 24.9 |
| `smallbatch` | 750 | 4 | 6 | 0.5 | 750 | ≈ 12.1 |
| `largebatch` | 150 | 20 | 30 | 0.5 | 150 | ≈ 61.3 |
| `fd80` | 375 | 8 | 12 | **0.8** | 375 | ≈ 29.6 |

- **Known label defect, non-behavioural.** Every `run_summary.json:/generalized/cardinality_sampler`
  carries the GENERALIZED-V1 record (`generalized_cardinality_uniform_v1`). This is a summary label
  only; no archived artifact is rewritten
  ([artifacts and metrics §6.1](../contracts/artifacts_metrics.md#61-known-summary-label-defect-run_summaryjsongeneralizedcardinality_sampler)).

### 8.4 Final primary endpoint

The endpoint is `SEVERE − MILD` aggregate `P(SELF_PRESERVATION_ABORT)` in the semantically
selected final round, macro-averaged with equal weight over the ten base cells of matched V2
groups ([training and benchmarks §9](../contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation)).
It comes from `run_summary.json:/generalized/v2_benchmark/final_round_behaviour`. The final-round
mean of the PR #65 pair deltas reproduces each value exactly.

| Variant | Final round (updates) | `macro_mean_over_base_cells` | Base cells defined | MILD→SEVERE selected-action switches (directional / reverse) |
|---|---:|---:|---:|---:|
| actor-only R1 | 375 | `+2.0936131477355958e-07` | 10 / 10 | 0 / 0 |
| CTDE R1 | 375 | `-2.415850758552551e-07` | 10 / 10 | 0 / 0 |
| `smallbatch` | 750 | `-2.9802322387695314e-09` | 10 / 10 | 0 / 0 |
| `largebatch` | 150 | `-9.080488234758377e-05` | 10 / 10 | 0 / 0 |
| `fd80` | 375 | `-8.512288331985474e-08` | 10 / 10 | 0 / 0 |

**Development finding: none of the five final policies showed meaningful severity-conditioned
selected-action separation on the frozen development benchmark.** These are small
aggregate-probability differences that produced no final MILD↔SEVERE selected-action switches;
they are not evidence of behavioural separation.

### 8.5 The three CTDE diagnostic arms

Each arm changes one training-configuration axis away from CTDE R1 and holds the design,
backend, manifest, profile and seed fixed (§8.3).

- **`smallbatch`.** Question: do more frequent PPO updates on smaller batches (4 successful
  episodes and ≈ 12 transitions per update, 750 updates, same 3000-episode budget) reveal or
  stabilize severity-dependent learning? **Final result: no severity separation.**
  **Transient:** in the rounds at updates 200 and 250, all 20 MILD and all 20 SEVERE immediate-FD
  wakes selected `SELF_PRESERVATION_ABORT`, with zero MILD↔SEVERE selected-action differences.
  **This was a global conservative mode, not severity discrimination.** It had reverted by the
  next evaluated round.
- **`largebatch`.** Question: does a larger, lower-variance update (20 successful episodes and
  ≈ 61 transitions per update, 150 updates) improve severity discrimination? **Final result: no
  severity separation.** **Transient:** at update 30 the round-mean `SEVERE − MILD` aggregate
  `P(ABORT)` was ≈ `+0.02847057`, yet all 20 MILD and all 20 SEVERE wakes still selected
  `PLAN_COMPLIANCE`. A sizable aggregate-probability sensitivity appeared transiently without
  crossing the deterministic action boundary.
- **`fd80`.** Question: is insufficient fuel-damage exposure the explanation? Training
  `fuel_damage_probability` was raised from R1's 0.5 to 0.8. Training population: 541 CLEAN /
  1247 MILD / 1212 SEVERE successful episodes and 2459 applied FD events, against 1474 / 774 / 752
  and 1526 in both R1 arms. **Final result: severity separation remained effectively zero
  despite substantially more FD training exposure.**

### 8.6 Matched immediate-FD wake analysis

**Provenance.** Temporary review PR #65, approved extraction candidate
`d565174e4ecc25eb60a4dd021e1a20025f55f07f`. It is a standard-library read-only extraction from
the five runs' already-recorded `wake_decisions`. It is not a scientific execution: no replay,
checkpoint load, policy recomputation, BLADE or solver run. Its source hashes equal the PR #61,
#62 and #64 ledgers, and the sources were unchanged by extraction. The package is not intended
as permanent repository evidence; its quantitative findings are recorded here.

**Coverage.** 5 variants × 16 evaluation rounds × 20 matched development groups per round:
**1600 MILD/SEVERE pairs and 3200 immediate-FD wakes**. Every expected pair formed, every MILD and
SEVERE member had exactly one immediate-FD wake, and `extraction_summary.json:/anomalies` is
empty. The pairs include each run's `pre_update` round. **They are matched pair observations
across repeated evaluations of the same frozen development worlds, not 1600 independent
worlds.**

**What the actor sees** (measured code `ae42cb01…`,
[policy and CTDE §1](../contracts/policy_ctde.md#1-graph-observation-stage-3)):

- the ego's own `fuel_norm = current_fuel / max_fuel`, clipped;
- a per-task `reachable_by_ego` bit, currently the conservative round-trip placeholder
  `round_trip_cost ≤ budget·(1 − σ)` rather than remaining-route slack
  ([policy and CTDE §6](../contracts/policy_ctde.md#6-known-limitations-and-open-items));
- a per-task distance, `dist_to_ego_norm`, recorded as `task_distance_norm`: haversine distance
  over the fixed normalizer `theater_scale_km`, clipped to `[0, 1]`.

- **Selected action.** The selected meta-action differed in **0 / 1600** pairs. Totals were
  MILD: 1560 `PLAN_COMPLIANCE`, 40 `SELF_PRESERVATION_ABORT`; SEVERE: 1560 and 40. All 80
  `ABORT` wakes are the `smallbatch` rounds at updates 200 and 250. This is a development
  finding, not an independence claim.
- **Fuel signal.** Recorded `SEVERE − MILD ego_fuel_norm` had mean ≈ `-0.43648`, min ≈ `-0.48432`
  and max ≈ `-0.35513`; the statistics are identical in every variant. **The actor observation
  therefore contains a large, direct post-damage fuel difference.**
- **Reachability signal.** These comparisons are positional: the records do not independently
  certify that task-node order is identical across the two members, as the PR #65 manifest
  cautions. No pair had exact-equal `reachable_by_ego` vectors, so all 1600 differ in at least
  one entry. Over 8000 compared entries, MILD→SEVERE transitions were `1→0`: 4000, `0→1`: 0,
  `1→1`: 2880, `0→0`: 1120. Every pair had at least one `1→0`, with a mean of 2.5 and a maximum
  of 7 per pair. Each variant shows the same aggregate pattern (800 / 0 / 576 / 224). **This is
  positional-vector evidence, not an independently reconstructed task-identity proof.**
- **Distance signal.** Recorded `task_distance_norm` vectors were exactly equal MILD vs SEVERE
  in all 1600 pairs. That includes the 7 pairs whose wake ticks differ by one (all in group
  `A5-D0-w001`). Equality is physically unsurprising, because fuel damage changes fuel, not
  location. **Clipping is near-total at these wakes:** in every one of the 160 round × severity
  wake populations, 19 of 20 wakes had every task distance clipped, for a mean clipped fraction
  of `0.99375`. This limits how informative absolute distance can be for route-relative
  reasoning. It does not establish that distance clipping caused anything.

### 8.7 Action-geometry finding

This is an observed structural phenomenon. **It is not a proven cause of the failure to learn
severity separation.**

The action distribution is a flattened `k × 3` categorical. Deterministic selection is
`torch.argmax` over the row-major joint cells. `PLAN_COMPLIANCE` and `SELF_PRESERVATION_ABORT`
keep node-indexed selection identities even though `PLAN` makes no node-scoped plan edit and
`ABORT`'s effect is ego-global and independent of the selected node
([policy and CTDE §2–§3](../contracts/policy_ctde.md#2-encoder-action-head-and-selection-stage-4)).

In the `smallbatch` rounds at updates 200 and 250, **for each severity separately**:

| Update | Selected `ABORT` | Aggregate meta-action argmax | `joint_vs_aggregate_disagree` | Mean aggregate `P(PLAN)` / `P(ABORT)` |
|---:|---:|---|---:|---|
| 200 | 20 / 20 | `PLAN` 17, `ABORT` 3 | 17 / 20 | ≈ 0.6577 / 0.3423 |
| 250 | 20 / 20 | `PLAN` 17, `ABORT` 3 | 17 / 20 | ≈ 0.6427 / 0.3573 |

In both rounds, deterministic joint-cell selection chose `ABORT` in all 20 wakes, while in 17 of
them most of the aggregate semantic mass lay on `PLAN`. **This is direct evidence that
joint-cell deterministic selection and aggregate semantic meta-action preference can diverge
materially under the current action representation.** On its own it does not establish that
this aliasing caused the absence of severity separation.

### 8.8 Transient aggregate severity sensitivity

The actor was not mathematically invariant to severity. The largest round-mean
`SEVERE − MILD` aggregate `P(ABORT)` over all 80 rounds were:

| Rank | Variant, update | Round mean |
|---:|---|---:|
| 1 | `largebatch`, 30 | `+0.028471` |
| 2 | `largebatch`, 70 | `+0.013429` |
| 3 | actor-only R1, 175 | `+0.006877` |
| 4 | `largebatch`, 80 | `+0.006050` |
| 5 | `largebatch`, 60 | `+0.005556` |

In all five rounds, MILD and SEVERE each selected `PLAN_COMPLIANCE` in 20 / 20 wakes, and
`joint_vs_aggregate_disagree` was false on every wake. **The learned distribution was capable of
transient severity sensitivity, but it did not become stable, severity-conditioned
deterministic behaviour.** Non-claim: the actor is not described as unable to distinguish
the states at all.

### 8.9 Actor-only versus CTDE

- Actor-only R1 and CTDE R1 both ended with effectively zero primary separation (§8.4). **No
  development evidence establishes a CTDE benefit for the target severity-conditioned
  behaviour**, and the three CTDE batch and exposure variants did not establish one either.
- The narrow conclusion: **the implemented centralized critic / GAE path was not sufficient to
  produce the observed target behaviour under these development runs.** This is not a general
  statement about CTDE.
- Credit semantics at the measured code: actor-only uses per-ego-grouped chains over the
  episode's scalar terminal reward (`compute_returns_and_advantages`). CTDE runs GAE over the
  episode's global decision sequence with a training-only centralized critic
  ([policy and CTDE §4](../contracts/policy_ctde.md#4-phase-b-ctde)).
- **No per-immediate-FD-transition MILD/SEVERE advantage diagnostic was persisted**, in either
  mode. Whether the relevant wake receives a stable, discriminative advantage signal
  **remains unresolved**.

### 8.10 Benchmark-provenance boundary and non-claims

- **Manifest identity is consistent.** All five run configs record `manifest_id
  ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`, profile `development`,
  `held_out_verified = true`. The external file is SHA-256
  `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103` at
  `C:/Users/Itama/PycharmProjects/graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0/benchmark_manifest.json`,
  as recorded by PR #62's `artifact_sha256.txt`; PR #64's `sweep/sweep_log.txt` records the same
  file hash on each arm's `ARM END` line.
- **Producer provenance (updated 2026-09-15; detail in §8.11).** The earlier statement here that
  the producer SHA was not known and producer provenance was open is superseded narrowly: the
  preflight's own report records producer code SHA `ae42cb01677f94868b2873008d87be677e31f0c8` on
  `main` with `dirty = false`, reviewed as producer-recorded provenance — not an external
  attestation, and not inferred from the directory name or the consuming runs' SHA. The exact
  original argv remains unknown; the historical research authorization and the historical prior
  review remain **not preserved / not proven**.
- Three things are therefore kept distinct: run, accounting and artifact consistency
  (established); the development interpretation, conditioned on the recorded frozen manifest
  (this section); and benchmark provenance — manifest identity reviewed, producer-recorded exact
  SHA and clean state reviewed, invocation partial, historical authorization and historical prior
  review not proven, current evidence review complete. **Full historical provenance is not
  complete.**
- The CTDE R1 verdict provenance in §7 is unchanged.
- **Non-claims.** No confirmatory result exists. No hypothesis is shown mathematically
  impossible. None of these is established as a cause of the missing severity separation: the
  action-geometry mismatch (§8.7), distance clipping, the reachability changes, or critic
  diagnostics. No CTDE conclusion beyond §8.9 is drawn.

### 8.11 GENERALIZED-V2 benchmark-preflight provenance review

Recorded on 2026-09-15 from temporary review PR #67 (branch
`review/v2-benchmark-preflight-provenance`, draft, exact reviewed candidate
`7f56338cde6aacfa59a52399b2378b98a62ea3aa`). GPT verdict on that exact candidate:
**`APPROVE — provenance package correctness / evidence review`**. The approval covers manifest
byte identity, the producer-recorded exact Git SHA, the producer-recorded historical clean state,
static manifest integrity, package and ledger correctness, and the separation of known from
unknown provenance. **It is not a scientific-validity approval, not a retrospective research
authorization and not proof of any prior historical review.** PR #67 is review transport, not
intended for merge; this section is the durable record. Building and reviewing the package
executed no project code — file reads, JSON parsing and SHA-256 hashing only.

- **Manifest identity — reviewed.** External local source
  `C:\Users\Itama\PycharmProjects\graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0`.
  `benchmark_manifest.json`: 204 244 bytes, file SHA-256
  `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`; `manifest_id
  ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`, recomputed over the
  canonical JSON of the record without `manifest_id` and equal to the stored id; schema
  `generalized_v2_benchmark_manifest`, version 1; required backend `p1_milp_v1`. **120 world
  groups / 360 members** (`clean`, `mild`, `severe` per group) over ten base cells `A2..A6` ×
  `D0/D2`, 12 worlds each; **development ordinals 0–1: 20 groups / 60 members; confirmatory
  ordinals 2–11: 100 groups / 300 members.** The report's `/manifest` block (id, file SHA-256,
  `seed_list_sha256 6d65318a5d44046b58bf44882d435c87172145d3cb4d3d21f1f31fe93fe88518`, 120 / 360)
  matches. Profile membership is a property of the manifest: **no confirmatory evaluation has
  been run.**
- **Producer code provenance — reviewed, producer-recorded.** From
  `benchmark_preflight_report.json:/provenance/git`: `available = true`, `commit =
  ae42cb01677f94868b2873008d87be677e31f0c8`, `branch = main`, `dirty = false`, `dirty_path_count
  = 0`, `repo_root = C:\Users\Itama\PycharmProjects\Multi_Agent_Task_Allocation_and_Adaptation`.
  This is **producer-recorded exact provenance, independently reviewed from the preserved
  artifact**: it was emitted by the executing project code (`_git_provenance(_REPO_ROOT)` in
  `graph_train.py`, called from `_run_v2_benchmark_preflight` in
  `graph_benchmark_preflight.py`, both checked at `ae42cb0…`). **It is not an external
  attestation**; it rests on that code having run as committed and on the report bytes being
  unaltered. The exact SHA is **not** inferred from the directory suffix `ae42cb0` or from the
  manifest `notes` text. The historical clean state is likewise only the report's own `dirty =
  false` (`git status --porcelain`: tracked changes and untracked non-ignored files); nothing is
  inferred from any later checkout.
- **Invocation provenance — partial.** *Known:* the effective request (`worlds_per_cell = 12`,
  `benchmark_base_seed = 2000000`, `max_candidates_per_cell = 64`, `n_base_cells = 10`) and
  settings (policy `deterministic_per_cell_window_fail_closed_v2`, design `generalized_v2`,
  backend `p1_milp_v1`, seeded-variable fuel damage, geometry) from the report; completion
  (`status = complete`, `manifest_written = true`, `failure = null`, 120 candidates attempted and
  accepted, 0 rejected); native exit code `0` (`native_exit_code.txt`); timestamps
  (`generated_utc = 2026-09-13T12:26:07.718521+00:00`; `invocation_start_local.txt` start
  15:25:53.69, end 15:26:08.26 local); environment clues the artifacts record (console warnings
  from the LOCAL Windows `nlp_env` site-packages; the Windows `repo_root`). *Unknown:* the exact
  original argv; whether `--config` was explicitly supplied; the exact Python version; how
  `label` / `notes` were supplied. A code-derived guess at the invocation is **not** recorded as
  fact.
- **Authorization and prior review.** Historical research authorization: **`NOT PRESERVED / NOT
  PROVEN`**. Historical prior review: **`NOT PRESERVED / NOT PROVEN`**. The manifest `notes`
  text containing "authorized" is operator-supplied descriptive text, not an authorization
  record. The PR #67 evidence review does not retroactively prove a historical review.
- **Technical conformance.** The effective persisted settings are descriptively consistent with
  the implemented GENERALIZED-V2 benchmark contract
  ([training and benchmarks §9](../contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation)).
  Technical conformance does not prove historical research authorization or scientific validity.
- **Provenance and reporting caveats** (source artifacts are not rewritten): the console banner
  prints the V1 preflight policy label `deterministic_per_cell_window_v1`, while the V2 report
  records the correct `deterministic_per_cell_window_fail_closed_v2`; the `START_UTC` label in
  `invocation_start_local.txt` holds local-time values; the exact argv is absent.

| Provenance item | State |
|---|---|
| Manifest identity | reviewed |
| Producer-recorded exact code SHA | reviewed |
| Producer-recorded clean state | reviewed |
| Invocation | partial |
| Historical research authorization | not preserved / not proven |
| Historical prior review | not preserved / not proven |
| Current evidence review (PR #67 @ `7f56338c…`) | complete |

Full historical provenance is **not** complete. Future confirmatory use of this manifest is
governed by the 2026-09-15 decision entry
([`decisions.md` §1](decisions.md#1-decision-log)).

## 9. Local artifact archive closure

Recorded on 2026-09-15 at the research-chapter Git and worktree cleanup
([`environments_cleanup.md` §4.7](../workflows/environments_cleanup.md#47-cleanup-already-performed)).
The archive was organized earlier the same day by authorized same-volume directory renames (move
timestamps in the ledger); the cleanup task re-verified it read-only — file reads and SHA-256
hashing only, no project code executed. **The dated records above keep their locations as
recorded**; this section adds current locations without rewriting them.

### 9.1 Archive identity

- Archive root `C:\gra\`; **28 indexed items**, each with `archive_result = MOVED_VERIFIED`.
- Machine-readable index `C:\gra\metadata\ARTIFACT_INDEX.jsonl` — 28 rows, SHA-256
  `de96d9ba4c04e10c075549d37a1b445a6e513d133ef55591a1849bd0b0b80552`; authoritative.
- Human-readable projection `C:\gra\metadata\ARTIFACT_INDEX.md` — SHA-256
  `a34d69a52c7f99f93abf402e516345f6c2c9eca0fa213f7bf6efb8c8a5f211c6`; not an independent source.
- Move ledger `C:\gra\metadata\ARCHIVE_MOVE_LEDGER.jsonl` — 28 rows, SHA-256
  `15287e8fa7b1051f48cd2b4d1d629f61d687c567d0c4858c5248569d8b6f9eb7`; per item: source,
  destination, `same_volume = true`, pre- and post-move file count, total bytes and key-file
  SHA-256 values.
- **Same-volume rename migration:** no artifact bytes were copied or rewritten, and no scientific
  artifact was deleted.
- **Embedded historical paths are unchanged and stale by design.** `output_dir`, `repo_root`,
  evidence-ledger, `RUN_IDENTITY` and script paths inside the artifacts still name the original
  locations; the index is the resolver. The index's own `evidence_ref_status` strings are
  archive-time text and still describe #61, #62, #64 and #67 as open
  ([`environments_cleanup.md` §4.4](../workflows/environments_cleanup.md#44-preserved-run-directories-and-external-artifacts)).

### 9.2 Preserved identities

Re-verified on 2026-09-15, before any Git evidence was removed: for **all 28 rows**, every indexed
key-file SHA-256 matched the bytes at the current path, and file count and total bytes matched
the index. For `v2_ctde_r1…` and the three diagnostic arms the counts exclude the added
`sidecars/` child, as the ledger's `post_count_note` states; each sidecar file's SHA-256 matched
the value in its row's caveats. The #61, #62, #64 and #67 evidence ledgers were also cross-checked
against the archive before their branches were deleted
([`environments_cleanup.md` §4.3](../workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs)).
The V2 preflight spot check: `benchmark_manifest.json`
`dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`,
`benchmark_preflight_report.json`
`a4c90680badb3ee12bae92fc13ae528ba85f00e26e59f9ed36493be2f6d4e216`, 120 scenario files.

"Measured SHA" is the value the index records, with its provenance and caveats in the index row;
it is not a verdict. Paths are relative to `C:\gra\`.

| Artifact id | Group | Current path | Measured SHA (index) | Files | Bytes | Key hashes |
|---|---|---|---|---|---|---|
| `v2_benchmark_preflight_seed2000000_ae42cb0` | benchmarks | `benchmarks\v2_preflight_seed2000000_ae42cb0` | `ae42cb0` | 125 | 2652472 | 125 |
| `v2_actor_only_r1_seed3000000_ae42cb0` | development | `runs\development\v2_actor_only_r1_seed3000000_ae42cb0` | `ae42cb0` | 3994 | 170956871 | 8 |
| `v2_ctde_r1_seed3000000_ae42cb0` | development | `runs\development\v2_ctde_r1_seed3000000_ae42cb0` | `ae42cb0` | 3993 | 185228275 | 7 |
| `v2_ctde_smallbatch_seed3000000_ae42cb0` | diagnostics | `diagnostics\v2_ctde_smallbatch_seed3000000_ae42cb0` | `ae42cb0` | 3993 | 185316529 | 25 |
| `v2_ctde_largebatch_seed3000000_ae42cb0` | diagnostics | `diagnostics\v2_ctde_largebatch_seed3000000_ae42cb0` | `ae42cb0` | 3993 | 184081905 | 25 |
| `v2_ctde_fd80_seed3000000_ae42cb0` | diagnostics | `diagnostics\v2_ctde_fd80_seed3000000_ae42cb0` | `ae42cb0` | 3993 | 190072468 | 25 |
| `v2_ctde_sweep_driver_ae42cb0` | diagnostics | `diagnostics\v2_ctde_sweep_driver_ae42cb0` | `ae42cb0` | 14 | 49921 | 14 |
| `phase_a_rerun_737b4bf` | valid | `runs\measurements\phase_a_rerun_737b4bf` | `737b4bf` | 5457 | 7543993559 | 8 |
| `phase_a_first_long_c30b698` | invalid | `runs\legacy_measurements\phase_a_first_long_c30b698` | `c30b698` | 5342 | 6826059323 | 9 |
| `probe_20260815_238062d` | invalid | `runs\legacy_measurements\probe_20260815_238062d` | `238062d` | 126 | 132861584 | 5 |
| `probe_20260816_900ff0b` | invalid | `runs\legacy_measurements\probe_20260816_900ff0b` | `900ff0b` | 126 | 160394934 | 6 |
| `b4_probe_a3f0838_unconfirmed_registry_match` | invalid | `runs\legacy_measurements\b4_probe_a3f0838_unconfirmed_registry_match` | `a3f0838` | 19 | 1713678 | 6 |
| `b4_observability_smoke_20260801_152146` | engineering | `legacy\engineering_smoke\training_output_b4_observability_smoke_20260801_152146` | — | 10 | 1562127 | 5 |
| `b4_observability_smoke_20260801_173605` | engineering | `legacy\engineering_smoke\training_output_b4_observability_smoke_20260801_173605` | — | 10 | 1562127 | 5 |
| `fd_variable_severity_valid_bf1e045f` | valid | `runs\measurements\fd_variable_severity_valid_bf1e045f` | `bf1e045` | 3201 | 4664082617 | 12 |
| `fd_variable_severity_invalid_precursor_bf1e045f` | invalid | `runs\legacy_measurements\fd_variable_severity_invalid_precursor_bf1e045f` | `bf1e045` | 3131 | 4181116530 | 6 |
| `generalized_v1_r1_4af6c5a` | valid | `runs\measurements\generalized_v1_r1_4af6c5a` | `4af6c5a` | 3993 | 157499888 | 36 |
| `p1_aborted_8f0d250_DO_NOT_RESUME` | invalid | `runs\legacy_measurements\p1_aborted_8f0d250_DO_NOT_RESUME` | `8f0d250` | 472 | 14508617 | 8 |
| `p1_fresh_ae194103` | valid | `runs\measurements\p1_fresh_ae194103` | `ae19410` | 3929 | 210665675 | 13 |
| `p1_fresh_preflight_revalidation` | review | `reviews\p1_fresh_preflight_revalidation` | — | 29 | 317640 | 3 |
| `p1_construction_audit_ae194103` | review | `reviews\p1_construction_audit_ae194103` | — | 95 | 35514071 | 17 |
| `phase_a_rerun_737b4bf_gpt_review` | review | `reviews\phase_a_rerun_737b4bf_gpt_review` | `737b4bf` | 1203 | 64282536 | 6 |
| `phase_a_rerun_737b4bf_gpt_review_zip` | review | `reviews\long_baseline_rerun_737b4bf_gpt_review.zip` | `737b4bf` | 1 | 3076694 | 1 |
| `task5a_engineering` | engineering | `legacy\engineering\task5a` | `09eab06` | 59 | 2929321 | 4 |
| `task5b_engineering` | engineering | `legacy\engineering\task5b` | `4af6c5a` | 133 | 3967760 | 4 |
| `ct1_possible_old_fixed_cell_ctde` | unknown | `legacy\unclassified\ct1_possible_old_fixed_cell_ctde` | — | 3202 | 4671039030 | 4 |
| `legacy_rollouts` | unknown | `legacy\unclassified\rollouts` | — | 35 | 497740 | 1 |
| `legacy_generated_scenarios` | unknown | `legacy\unclassified\generated_scenarios` | — | 4 | 97663 | 1 |

Group labels are archive organization, not verdicts; every verdict stays as recorded in §1–§8.

### 9.3 Recovered local identities

Recorded to close earlier "not recorded" gaps where the archived bytes support it, and only that
far. None changes a verdict.

- **First real post-B3 probe (`a3f0838`).** The archived directory formerly named
  `training_output_b4_probe_20260730_182528` — now
  `runs\legacy_measurements\b4_probe_a3f0838_unconfirmed_registry_match` — has **6 of 6** key
  files (`run_config.json`, `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl`,
  `run_summary.json`, `training_plot.png`) byte-identical to the evidence SHA-256 values of the
  `a3f0838` record in §2, re-hashed on 2026-09-15. This is an **archive-time identity match to the
  recorded first post-B3 probe**, which strongly closes the registry's location gap. No claim is
  made about where the directory was when the probe was reviewed; the archive directory name is
  kept as it is.
- **Aborted P1 arm.** The archived arm's own `RUN_IDENTITY.txt` and `run/run_config.json` record
  measured code SHA `8f0d250cd9f96e6b8bce635065701dc47a5ee87e` (`run_config.json`: `dirty =
  false`; `RUN_IDENTITY.txt`: detached execution worktree `C:/p1src`, backend `p1_milp_v1`,
  `actor_only`, `generalized_v1`). This is **recovered local artifact provenance for the
  already-invalid arm of §6.1**: it stays **`ABORTED / DO NOT RESUME`**, is not a measurement and
  is not upgraded by this identity. Current path
  `runs\legacy_measurements\p1_aborted_8f0d250_DO_NOT_RESUME`.
- **Task 5A.** The preserved engineering artifact's `README_ENGINEERING_ONLY.txt`, `console.log`
  and `r/run_config.json` (`dirty = false`) record repository SHA
  `09eab0673153bd443185ec94530ccf0b042be465`, a commit
  reachable from `main`. This closes the missing local Task 5A identity of §1; it remains
  **engineering validation only**, not a measurement. Current path `legacy\engineering\task5a`.
- **GENERALIZED-V1 R1.** The run tree, formerly `C:\g1r1`, is now
  `runs\measurements\generalized_v1_r1_4af6c5a`; among its 36 re-verified key files is
  `GENERALIZED_V1_R1_FD_POLICY_DIAGNOSTICS.zip` with the recorded bundle SHA-256
  `812ff43322e134e9a7ca31720007393ff1220ba50c35955b2a724b30d4d5d792`.
- **Fresh deterministic-P1 arm.** Formerly `C:\p1_fresh_ae194103`, now
  `runs\measurements\p1_fresh_ae194103`; its index row records the measured SHA from the arm's
  own review bundle, equal to §6.2's `ae1941035991df4719df212c4b5dd07db89aee4a`. §6.2's statement
  that no identity, denominators or artifacts are recorded stays as the record it was; nothing
  beyond location and key hashes is added here.
- **Fresh P1 construction audit.** Beyond its directory name, the preserved
  `results\construction_audit_report.json` itself records `repository_sha` and `head_sha`
  `ae1941035991df4719df212c4b5dd07db89aee4a` with an empty `porcelain` on `main` — provenance
  emitted by the audit tooling, not an external attestation. It remains **engineering / design
  evidence**, not a measurement. Current path `reviews\p1_construction_audit_ae194103`.
- **`ct1`.** `legacy\unclassified\ct1_possible_old_fixed_cell_ctde` (formerly
  `C:\Users\Itama\ct1`) records `run_config` commit `76abdc480e80a84f1503208730d4525cd5e89b69`
  and `training_mode = ctde`. **That is not sufficient to identify it as the old fixed-cell CTDE
  measurement of §1.** Classification stays **`UNKNOWN / possible old fixed-cell CTDE arm`**; it is
  not reinterpreted, reviewed or compared.

### 9.4 Non-claims

No artifact under `C:\gra\` was modified or deleted by the cleanup; no verdict, measured SHA or
classification in §1–§8 is changed; no invalid, aborted, engineering or unclassified artifact is
upgraded; the B4 engineering smokes, `rollouts` and `generated_scenarios` stay uninterpreted;
closing the evidence PRs and deleting their branches removes no conclusion recorded in this
document.

## 10. GENERALIZED-V2 semantic-action actor-only development R1

Recorded on 2026-09-16. The decisions taken on this record are in
[`decisions.md` §1](decisions.md#1-decision-log) (the 2026-09-16 rows that follow the
implementation-opening row). §7–§9 stay as the records they were.

### 10.1 Scope and epistemic status

| Layer | What it is | Status |
|---|---|---|
| measurement validity | the run executed its recorded authorized plan, completed and reconciled | **`APPROVE — VALID DEVELOPMENT MEASUREMENT`** — GPT verdict of 2026-09-16 on the exact evidence candidate below, as transferred in the user-approved documentation packet; GitHub holds no separate review record (§10.2–§10.3) |
| observed findings | final endpoint, evaluation trajectory, credit summaries | reproduced from the evidence package (§10.4–§10.6) |
| interpretation | what the findings are consistent with | a **development** interpretation, cross-version against historical actor-only R1; hypotheses, not proofs (§10.8–§10.9) |
| confirmatory evidence | — | **none**. The confirmatory profile was not selected, inspected or executed |

**Repeated measures.** Every evaluation round re-measures the same 20 frozen development worlds;
cross-round values describe a trajectory, never independent samples
([`experiments.md` §4.3](../workflows/experiments.md#43-interpretation-rules)).

### 10.2 Identity and evidence provenance

| Item | Value |
|---|---|
| Run id | `graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37` |
| Measured code SHA | `d4e9f3721e6d151c00be3fe93c3d149df9d31965` (the PR #70 merge); run-recorded branch `main`, `dirty = false`, 0 dirty paths |
| Training mode | `actor_only` |
| Action representation | `semantic_k_plus_2_logmeanexp_v1` — uniform on every episode outcome, wake and credit row |
| Population and benchmark | `generalized_v2`, `p1_milp_v1`, base seed `3000000`; manifest `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8` (file SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`), **development** profile, held-out verified — the same frozen development population and the same training seed stream as historical actor-only R1 (§7) |
| Budget | 375 updates × 8 successful episodes; `generalized_max_attempts_per_iteration = 12`; `eval_every = 25`; `checkpoint_every = 25`; **early stopping disabled** (`disabled_fixed_budget`) |
| PPO (resolved) | `gamma = 1.0`, `lr = 0.0003`, `clip_ratio = 0.2`, `entropy_coeff = 0.01`, `n_epochs = 4`, `max_grad_norm = 0.5`, `adv_norm_eps = 1e-08` |
| Authorized plan | `authorized_plan.json` (SHA-256 `629603a717b9933f7f956c08ef1a802556f6c0076c77bb8bee99ede05e71a467`), written before `run_config.json`; one arm, `ctde_arm_authorized = false`, `hyperparameter_sweep_authorized = false`; every checked resolved field matches the plan and the invocation argv matches it |
| Evidence | draft PR #71, branch `evidence/generalized-v2-semantic-action-actor-only-dev-r1`, **exact candidate `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`** (parent chain `bf34e955…` → measured SHA); package `research_evidence/generalized_v2/semantic_action_actor_only_dev_r1/`; **not for merge** |
| Large artifacts | outside Git, identified by SHA-256 in the package's `artifact_sha256.txt`: `episode_outcomes.jsonl` `3011a163b3b341d5ab25881dae2394e45d7a5ae8ec66854a3f5088d147198740`, `train_credit_diagnostics.jsonl` `6f04ff38ec7beb5009b85c9d35a97527369c5f83257b3590e58e7e904d06b72e`, final checkpoint `ckpt_iter0374.pt` `c231b812c1e19376e5888c9d426c0a1c3a487c41144bda9a829aa622c8bde1f6` |
| Comparator | historical actor-only R1, measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`, archive key `v2_actor_only_r1_seed3000000_ae42cb0`; its per-round quantities were **recomputed from the archived artifacts** by the package's extractor, not taken from prose |

Evidence-commit SHAs are ledger locations, not measurement identities
([artifacts and metrics §6.2](../contracts/artifacts_metrics.md#62-measured-code-sha-versus-evidence-commit-identity)).

### 10.3 Validity facts

- **Completion:** 375 / 375 updates; `train_records.jsonl` has 375 rows; final checkpoint present.
- **Training accounting:** 3000 successful episodes of 3008 attempted; **8 failed and replaced**,
  all `FuelDamageError` (`no_fd_eligible_ego`) at phase `train`, stage `setup` — the expected
  certified-FD eligibility attrition class. The failed seeds (`3001255`, `3001265`, `3001741`,
  `3001807`, `3001879`, `3002177`, `3002296`, `3002868`) are **exactly the failed seeds of
  historical actor-only R1**. `accounting_reconciled = true`.
- **Evaluation:** 16 rounds; 960 / 960 evaluation episodes successful, 0 failed.
- **Endpoint eligibility:** the final primary endpoint is defined on 10 / 10 base cells;
  20 / 20 groups complete and metric-eligible.
- **Credit coverage:** 9166 credit rows cover every transition of every productive update
  (375 / 375 updates' row counts equal `train_records.n_transitions`).
- **Schemas:** uniform episode outcome v4, wake diagnostics v2, credit diagnostics v1.
- **Configuration:** the resolved training configuration differs from historical R1 only in
  `benchmark_manifest` (path) and `output_dir`; the action representation is a code difference
  at the measured SHA, not a configuration key.

**Non-blocking anomalies** (recorded, not repaired; source artifacts are not rewritten):

1. `native_exit_code.txt` is empty (0 bytes): the launcher line `echo %RC%> file` is parsed by
   `cmd` as a handle redirect for a one-digit code. Completion is established independently by
   `run_summary.json`, `train_records.jsonl` and the final checkpoint.
2. `v2_behaviour.metric` reads the legacy string `severe_minus_mild_aggregate_abort_mass`. The
   same block defines `P(ABORT)` as the one semantic `SELF_PRESERVATION_ABORT` leaf and sets
   `aggregate_mass_is_not_selected_action_probability = false`. This is the contracted
   reader-continuity label ([training and benchmarks §9](../contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation)),
   not a value defect.
3. The manifest was consumed at its authorized archived path
   `C:\gra\benchmarks\v2_preflight_seed2000000_ae42cb0\benchmark_manifest.json`, while the
   comparator recorded the pre-archival path; the file is byte-identical to the historical source
   (same SHA-256 and `manifest_id`), so this is not a population change.

### 10.4 Final primary endpoint

Final round selected semantically: `post_update`, 375 updates, evaluation round 15.

| Quantity | Value |
|---|---|
| Ten-cell macro `SEVERE − MILD P(ABORT)` | `+0.000888290349394083` |
| Base cells defined | 10 / 10 |
| Metric-eligible groups | 20 / 20 |
| MILD → SEVERE directional switches | 0 / 20 |
| Reverse switches | 0 / 20 |

**The run does NOT establish stable final severity-conditioned behaviour.** The final value is
numerically non-zero but effectively zero, with no selected-action switch in any group; it is not
a success.

### 10.5 Transient learned separation

Key trajectory (ten-cell macro; directional switches out of 20 groups; reverse switches were
0 / 20 in every round):

| Updates | Macro SEVERE − MILD P(ABORT) | Directional switches |
|---:|---:|---:|
| 75 | `+0.2955018974840641` | 4 / 20 |
| 100 | `+0.6111742591485381` | 19 / 20 |
| 125 | `+0.6502656679600477` | 20 / 20 |
| 150 | `+0.5764810834079981` | 19 / 20 |
| 175 | `+0.0028667372651398184` | 0 / 20 |

Rounds at updates 0–50 lie within `[−0.00052, +0.000011]`, and every later round (updates
175–375) stays small, between `+0.00045` and `+0.0061`, with 0 directional switches, ending at
the final value of §10.4. All 16 rounds, with per-cell and per-group values, are in the
package's `extracted/behaviour_summary.json` and `extracted/collapse_timeline.json`.

**Historical actor-only R1 comparator** (measured SHA `ae42cb01677f94868b2873008d87be677e31f0c8`,
recomputed from its archived artifacts): largest observed round `+0.006876615434885025` (updates
175); **0 directional switches in every round**; final macro `+2.0936131477355958e-07`.

**Comparison boundary.** This is a **cross-version DEVELOPMENT comparison** — different measured
code SHA and action representation, same frozen development population and training seed
stream. It is **not a contemporaneous randomized control and not confirmatory evidence**.

### 10.6 Credit findings

Population: training credit rows with `wake_kind = immediate_fuel_damage` and
`is_fd_selected_ego = true` — **1526 rows: 774 MILD, 752 SEVERE**.

**Descriptive severity summaries** (pooled over all training updates; different episodes, worlds
and policy states):

| Severity | n | Raw advantage mean | Normalized advantage mean |
|---|---:|---:|---:|
| MILD | 774 | `+0.09266661183010043` | `+0.372028710830354` |
| SEVERE | 752 | `-0.32571354607250536` | `-1.146134736059606` |

**Matched within update batch** (309 batches contain both severities; same baseline and
normalization, still different episodes and worlds):

| `SEVERE − MILD` | Mean | Median |
|---|---:|---:|
| raw advantage | `-0.4715824457157431` | `-0.4062499894748267` |
| normalized advantage | `-1.8726210434928396` | `-1.9523004768366192` |

SEVERE lies below MILD in raw advantage in **299 / 309** batches.

**Within-severity ABORT-vs-not** (rows that selected ABORT versus rows that did not):

| Severity | Raw ABORT − not gap | Normalized ABORT − not gap |
|---|---:|---:|
| SEVERE (202 ABORT / 550 not) | `+0.20462639823500103` | `+0.17649193803586716` |
| MILD (64 ABORT / 710 not) | `-0.3048436088994812` | `-1.508242941125282` |

**These are NOT counterfactual action-value estimates.** They compare different episodes and
worlds.

### 10.7 Structural credit limitation

**This is a major reviewed finding.** Verified from the persisted credit rows, not only from code
(`graph_ppo.compute_returns_and_advantages` and `_chain_returns` at the measured SHA):

- `gamma = 1.0` in `run_config.json` and on every row;
- **6689 ego chains checked, 0 with varying raw advantage** (1222 have more than one transition);
- **2999 episodes checked, 0 with varying raw advantage** (2415 have more than one transition);
- in **2999 / 2999** episodes the stored transition rewards sum to the episode reward;
- **1114** episodes have a non-zero reward, and in each all of it sits on exactly one transition,
  at the episode's latest tick and last in its ego chain;
- per row: `return == episode_reward`, `raw_advantage == return − baseline`, and the normalized
  advantage recomputes from the batch moments; the baseline equals `train_records.baseline`
  (recomputable from rows in 374 batches; one batch's zero-wake episodes contribute no row).

**Actor-only credit is therefore episode / chain-level, not local causal credit for the
immediate-FD decision.**

> The immediate-FD transition's actor-only advantage is effectively the episode outcome relative
> to the batch baseline. Severity differences therefore partly restate different episode
> outcomes. The observed SEVERE ABORT-vs-not advantage gap is suggestive association, not proof
> that ABORT caused the better outcome.

### 10.8 Research interpretation

The reviewed current interpretation (development only):

1. **Actor-visible severity signal was already known to exist** (§8.6).
2. **The old action geometry was a demonstrated structural mismatch** (§8.7).
3. **With the semantic representation the actor can learn strong severity-conditioned behaviour
   temporarily:** strong separation emerged by update 75 (macro `+0.296`, 4 / 20 directional
   switches), then became near-universal across the frozen development worlds at updates
   100–150 (19 / 20, 20 / 20 and 19 / 20 directional switches), before collapsing by update 175 —
   a qualitatively new development behaviour that historical R1 never showed.
4. **The behaviour is not retained to the fixed final budget:** it collapses by update 175 and
   the final endpoint is effectively zero.
5. **Actor-only advantage is too coarse to identify local FD-action credit** (§10.7).
6. **The next focused question** is whether a richer state-dependent training critic / GAE path
   can preserve the learned separation under the semantic action representation.

Defensible reading: this is evidence that the historical action geometry was a **material
bottleneck / contributor**; the remaining research problem is now primarily **retention /
optimization / credit stability**, rather than the action space's inability to express the
desired behaviour.

### 10.9 Limitations and non-claims

- It is **not** standalone causal proof that action aliasing was the only cause of the historical
  failure; the comparison is cross-version with no contemporaneous control (§10.5).
- The action-representation change did **not** solve the project objective: the final endpoint
  is effectively zero.
- One run, one seed stream, 20 frozen development worlds re-measured every round; no variance
  across training seeds is estimated.
- Credit comparisons are descriptive or matched-within-batch associations, never action values
  (§10.6–§10.7).
- **No claim that CTDE will succeed.** Historical CTDE at `ae42cb0…` did not establish a benefit
  under the old representation (§8.9); CTDE under the semantic representation is unmeasured.
- Reachability (still the round-trip placeholder) and distance clipping (§8.6) were deliberately
  unchanged and remain open; this measurement does not address them.
- No confirmatory evidence exists; the confirmatory profile is untouched.

## 11. GENERALIZED-V2 semantic-action CTDE actor-gradient development diagnostics

Recorded on 2026-09-17. The decisions taken on these records are the 2026-09-17 rows of
[`decisions.md` §1](decisions.md#1-decision-log) that follow the instrumentation-opening row.
§10 stays as the record it was.

### 11.1 Scope and epistemic status

| Layer | What it is | Status |
|---|---|---|
| measurement validity | each run executed its recorded authorized plan, completed and reconciled | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** for both runs — GPT orchestrator review of 2026-09-17, as transferred in the user-approved documentation packet (§11.2–§11.3) |
| observed findings | gradient windows, held-out trajectory, credit and entropy quantities | reproduced from the original gradient, credit, evaluation and summary artifacts by the compact index's extractor (§11.4–§11.7) |
| development interpretation | what the findings are consistent with | **development only**; hypotheses, not proofs; no causal attribution (§11.8) |
| unresolved hypotheses and next research action | open mechanism questions | a read-only mechanism audit, not a conclusion that any item is defective (§11.9) |
| confirmatory evidence | — | **none**. The confirmatory profile was not selected, inspected or executed |

**Repeated measures.** Every evaluation round re-measures the same 20 frozen development worlds;
cross-round values describe a trajectory, never independent samples
([`experiments.md` §4.3](../workflows/experiments.md#43-interpretation-rules)).

### 11.2 Identity and evidence provenance

| Item | Run A — `p = 0.5` diagnostic baseline | Run B — FD100 intervention |
|---|---|---|
| Run id | `graph_rl_v2_semantic_ctde_grad_diag_r1_seed3000000_6ed964a` | `graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` |
| Local original (authoritative, external, untouched) | `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_r1_seed3000000_6ed964a` | `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` |
| Wall clock (local, 2026-09-17) | 16:02:53 – 16:55:49 | 17:34:17 – 18:19:05 |
| `fuel_damage_probability` / `fuel_damage_mild_probability` | `0.5` / `0.5` | **`1.0`** / `0.5` |
| Authorized plan SHA-256 | `d157f37345dc6486256896d25bc52b149e00bb39ae3afbfd68f7d8d4a9f11330` | `99571b95b09bc93021234d4675915720dbe9ec12b692ee3f94b816ffa3320cf0` |

Shared by both runs:

| Item | Value |
|---|---|
| Measured code SHA | `6ed964a1abd09de2130aee3d0d314c8f32165056`, branch `task/v2-ctde-gradient-pressure-diagnostics` — the GPT-approved head of PR #74, **unmerged when measured**; `run_config.json:/provenance/git` records `dirty = false`, 0 dirty paths |
| Design | GENERALIZED-V2, `training_mode = ctde`, `p1_milp_v1`, `semantic_k_plus_2_logmeanexp_v1`, DEVELOPMENT profile of manifest `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`, base seed `3000000` |
| Budget | 150 updates × 8 successful episodes; `generalized_max_attempts_per_iteration = 12`; `eval_every = 25`; `checkpoint_every = 25`; early stopping disabled |
| PPO / CTDE | `gamma = 1.0`, `lr = 0.0003`, `clip_ratio = 0.2`, `entropy_coeff = 0.01`, `n_epochs = 4`, `max_grad_norm = 0.5`, `adv_norm_eps = 1e-08`; `critic_lr = 0.0003`, `value_coeff = 0.5`, `gae_lambda = 0.95` — otherwise the frozen semantic CTDE configuration |
| Diagnostic | `--actor-gradient-diagnostics` on ([artifacts and metrics §5.2](../contracts/artifacts_metrics.md#52-ctde-actor-gradient-diagnostics)); training credit diagnostics as always ([§5.1](../contracts/artifacts_metrics.md#51-training-credit-diagnostics)) |
| Compact Git index | `research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/` on PR #74: README, `artifact_manifest.json`, `diagnostic_summary.json` and a standard-library extractor that reads the external originals (`--check` regenerates both JSON files byte-identically). **No record stream or checkpoint is committed** |

**Comparison boundary.** The only scientific configuration intervention in Run B versus Run A is
`fuel_damage_probability 0.5 → 1.0`, plus the required fresh `output_dir`. The flattened
`run_config.json` difference is exactly that key, its derived `difficulty` block (FD probability
and scheduled cell probabilities), the output directory, the invocation argv and the collection
timestamp. **Known BLADE run-to-run timing nondeterminism remains a comparison limitation**: the
two runs' physical executions are not claimed bit-identical apart from the intervention. The
evaluation population is unchanged (the forced CLEAN / MILD / SEVERE benchmark).

**External artifact identities** (SHA-256; all verified against the originals):

| Artifact | Run A | Run B |
|---|---|---|
| `authorized_plan.json` | `d157f37345dc6486256896d25bc52b149e00bb39ae3afbfd68f7d8d4a9f11330` | `99571b95b09bc93021234d4675915720dbe9ec12b692ee3f94b816ffa3320cf0` |
| `preflight.json` | `e7236413fc57735b0831cb7f02dad67bc2cb6ee8bcc8da13ba765d0fa83a9156` | `eb830d3b373847b84396d2da7579cb247eae2ddf43772b4581fefcfe07e000bd` |
| `run_config.json` | `a1129e6dec81ccf136b096987a057e4bb21676082a01c26d510a2be85d27dcf5` | `1ddeb5ed080a6af03dc8e71a3d5dba042258c580665d075b02a7d86d1ec0c38f` |
| `run_summary.json` | `29ca9e60cf5a0ca5af05d411ad6496468f1f8bb261ade30d277596b4aed931ab` | `dbfa3e9134ad93cd36e65f8409d74e7962540796341cf82e5d4b76ee2feb0bd1` |
| `train_records.jsonl` | `402c0fce5d67c623eb85edcbf4f7c00e2dec3856500dd5225a29e314e330f88c` | `d7637750081351cc55cb4505d28045f6fd7738cb2afbd841e08a342fb02d67ea` |
| `eval_records.jsonl` | `388b3e4b51b1d78e6ce80d4da90b27202dc79de86000e4d54b0b22481784e5f4` | `1117e44a0c92896f3338253f0e2a37a8fdf8b8504ff9660cdefa42daf2a54b82` |
| `episode_failures.jsonl` (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `train_actor_gradient_diagnostics.jsonl` | `e445f45ff551e1eb78cf83fddfe9926f527e1e6bef6ebbebea02c17946aa2aed` | `c29bbf8b4ac319f3091d906d6928a8559ec34a49ae7220e71e1c819a3ca39aec` |
| `train_credit_diagnostics.jsonl` | `2754c7ec6bb56b97fc0a0bfeb5c194945369af2301650ee5aa7858a925590ee0` | `87c0c3acadc08ea33fa33fb9f008df43de08bfc2307955b400a5177d60567a6b` |

`episode_outcomes.jsonl`, the console log, the launcher files, the six checkpoints and the
scenario and plot trees are identified in the index's `artifact_manifest.json`.

### 11.3 Validity facts

| Fact | Run A (`p = 0.5`) | Run B (FD100) |
|---|---|---|
| Productive updates | 150 / 150 | 150 / 150 |
| Training episodes successful | 1200 / 1200; 0 failures, 0 replacements | 1200 / 1200; 0 failures, 0 replacements |
| Trained transitions (`run_summary` = sum of `train_records.n_transitions`) | 3581 | 4514 |
| Training clean / damaged successful | 612 / 588 | **0 / 1200** |
| FD events applied / immediate-FD wakes | 588 / 588 | 1200 / 1200 |
| MILD / SEVERE training episodes | 302 / 286 | 625 / 575 |
| Evaluation rounds; episodes successful | 7; 420 / 420 | 7; 420 / 420 |
| Final round groups complete / metric-eligible | 20 / 20 / 20 | 20 / 20 / 20 |
| `accounting_reconciled` | `true` | `true` |
| Gradient diagnostic rows | 150, one per productive update | 150, one per productive update |
| Updates with both MILD and SEVERE immediate-FD transitions (defined contrast) | **120 / 150** | **149 / 150** |
| Credit diagnostic rows | 3581, exactly one per trained transition | 4514, exactly one per trained transition |
| Maximum gradient reconstruction relative error | `5.0e-7` | `1.0e-6` |

Run B is genuinely FD100 on the training distribution. Both `native_exit_code.txt` files hold
`0`, and neither console log contains `CRASH` or `Traceback`.

### 11.4 Observed gradient findings

Definitions ([artifacts and metrics §5.2](../contracts/artifacts_metrics.md#52-ctde-actor-gradient-diagnostics)):
a component's `separation_pressure = −dot(h, g)` is its local first-order raw-gradient push on
`mean P(ABORT | SEVERE) − mean P(ABORT | MILD)`; positive pushes toward larger separation. **FD
pressure** is `derived.fd`, **non-FD** is `derived.non_fd`, **total** is the total policy
surrogate. Windows and counts use only updates with a defined contrast; medians are over those
updates.

**Run A (`p = 0.5`).** FD separation pressure is positive in **56 / 120 = 46.7%** of defined
updates.

| Iterations | Defined | FD positive | Median FD pressure | Median non-FD pressure | Median total surrogate pressure |
|---|---:|---:|---:|---:|---:|
| 0–24 | 21 | 52.4% | `+0.003633` | `+0.012425` | `+0.027715` |
| 25–49 | 22 | 40.9% | `-0.000326` | `+0.006101` | `+0.002160` |
| 50–74 | 22 | 54.5% | `+0.002959` | `-0.000318` | `-0.001298` |
| 75–99 | 18 | 22.2% | `-0.007114` | `+0.000501` | `-0.009729` |
| 100–124 | 17 | 64.7% | `+0.000643` | `+0.000077` | `+0.001874` |
| 125–149 | 20 | 45.0% | `-0.000422` | `-0.000062` | `+0.001232` |

Only **19 / 120** defined updates have FD pressure `> 0` with non-FD pressure `< 0`, and only
**9 / 120** have a positive FD pressure that becomes a negative total surrogate pressure. **The
`p = 0.5` result therefore does NOT support simple non-FD cancellation as the dominant
explanation.** In the critical 75–99 window the FD component itself is negative in **14 / 18**
defined updates.

**Run B (FD100).** FD separation pressure is positive in **98 / 149 = 65.8%** of defined updates.

| Iterations | Defined | FD positive | Median FD pressure | Median non-FD pressure | Median total surrogate pressure |
|---|---:|---:|---:|---:|---:|
| 0–24 | 24 | 75.0% | `+0.014278` | `+0.009408` | `+0.013295` |
| 25–49 | 25 | 52.0% | `+0.000656` | `+0.002084` | `-0.000437` |
| 50–74 | 25 | 80.0% | `+0.019473` | `+0.018952` | `+0.053334` |
| 75–99 | 25 | 52.0% | `+0.000257` | `-0.000663` | `-0.000679` |
| 100–124 | 25 | 76.0% | `+0.000341` | `-0.000089` | `-0.000217` |
| 125–149 | 25 | 60.0% | `+0.000275` | `+0.001297` | `+0.000692` |

FD100 materially improves FD-gradient directional coherence, especially before update 75.
Cancellation / interference becomes visible after acquisition: **37 / 149** defined updates
have positive FD pressure and negative non-FD pressure, and in **23 / 149** a positive FD
pressure becomes a negative total surrogate pressure. **This is not described as the sole cause
of the later collapse.**

### 11.5 Behavioural findings

Held-out DEVELOPMENT evaluation trajectory, ten-cell macro `SEVERE − MILD P(ABORT)`; every round
has 60 / 60 episodes, 10 / 10 base cells defined, 20 / 20 groups metric-eligible, and **0 / 20
directional and 0 / 20 reverse selected-action switches in both runs**:

| Updates | Run A (`p = 0.5`) | Run B (FD100) |
|---:|---:|---:|
| 0 (pre-update) | `-0.0005158037` | `-0.0005158037` |
| 25 | `-0.0000049397` | `+0.0000906825` |
| 50 | `+0.0000051774` | `+0.0020783611` |
| 75 | `+0.0001370527` | **`+0.0427981213`** |
| 100 | `+0.0000016853` | `-0.0000086490` |
| 125 | `+0.0000046670` | `-0.0000035509` |
| 150 | `-0.0000008188` | `-0.0000019804` |

- **Run A:** no meaningful severity-conditioned behaviour appears.
- **Run B:** at update 75 all 20 matched groups are measurable, but there are still 0 / 20
  directional switches; the separation collapses by update 100 and stays effectively zero through
  update 150.

**FD100 improves acquisition of probability-level severity separation but does not produce stable
retained severity-conditioned behaviour.** It is qualitatively stronger than the `p = 0.5`
diagnostic run but much weaker than the semantic actor-only transient at updates 75–150 (§10.5).

### 11.6 Credit findings

Population: training credit rows with `wake_kind = immediate_fuel_damage` and
`is_fd_selected_ego = true` — Run A **588** (MILD 302: 251 PLAN / 51 ABORT; SEVERE 286: 229 PLAN /
57 ABORT); Run B **1200** (MILD 625: 485 PLAN / 140 ABORT; SEVERE 575: 443 PLAN / 131 ABORT /
1 OPPORTUNISTIC_ENGAGEMENT, which is outside the ABORT − PLAN comparison).

**Action-conditioned normalized advantage** (mean over ABORT rows minus mean over PLAN rows,
within severity, pooled over updates):

| Severity | Run A `ABORT − PLAN` | Run B `ABORT − PLAN` |
|---|---:|---:|
| MILD | `-1.326949` | `-1.118971` |
| SEVERE | `+0.165636` | `+0.392281` |

FD100 makes the descriptive SEVERE action-credit association more consistently favourable to
ABORT while preserving the negative MILD ABORT association. **These remain NON-COUNTERFACTUAL
comparisons across different sampled episodes and actions.**

**Critic decision-locality** — within-update `SEVERE − MILD` (per update holding both severities:
`mean(SEVERE) − mean(MILD)`; median over 120 updates in Run A, 149 in Run B):

| Quantity | Run A | Run B |
|---|---:|---:|
| `value_old` | `-0.000560` | `-0.000073` |
| `value_target` | `-0.351518` | `-0.353858` |
| raw advantage | `-0.330884` | `-0.357481` |

Increasing FD exposure does **NOT** make the critic's pre-update value estimate meaningfully
severity-sensitive at the immediate-FD decision, despite large target and advantage differences.

**Local TD residual relative to raw GAE advantage** (per-row median of
`|td_residual| / |raw_advantage|`): Run A MILD ≈ `0.48%`, SEVERE ≈ `0.19%`; Run B MILD ≈
`0.20%`, SEVERE ≈ `0.25%` — tiny in the median. **This ratio is a descriptive diagnostic, not a
formal decomposition of causal credit.** It is a median statement only: the ratio of mean
magnitudes is much larger (Run A `19.4%` / `12.8%`, Run B `17.6%` / `12.6%`, MILD / SEVERE), so a
minority of rows carry a substantial local residual.

### 11.7 Entropy finding

- **Run A:** total surrogate versus actual actor-loss gradients have mean cosine ≈ `0.999` over the
  120 defined updates (`0.993` over all 150 rows); **0 / 120** defined updates have the entropy
  term flip the sign of separation pressure.
- **Run B:** the mean cosine is not the load-bearing fact, because a few updates differ in
  magnitude / direction (minimum cosine `0.39`); the exact fact is that **0 / 149** defined updates
  have the entropy term flip the sign of total separation pressure.

**There is no current evidence that the entropy bonus explains the acquisition or retention
failure.**

### 11.8 Development interpretation

DEVELOPMENT ONLY. The reviewed supported interpretation:

1. The `p = 0.5` diagnostic does **not** support "a healthy FD gradient simply drowned by ordinary
   transitions" as the dominant mechanism: the FD component itself is frequently incoherent with
   the desired severity contrast (§11.4).
2. Increasing FD exposure to 100% materially changes the learning signal: almost every update
   contains both severities (120 / 150 → 149 / 150); FD separation pressure is positive more often
   (46.7% → 65.8%); the SEVERE ABORT-vs-PLAN normalized advantage becomes more favourable
   (`+0.166` → `+0.392`); held-out probability-level separation reaches `+0.0428` at update 75.
3. This is evidence that FD exposure is a **material contributor to acquisition**.
4. It does **not** solve **retention**: separation collapses by update 100 and is effectively zero
   thereafter.
5. More exposure does not make `value_old` meaningfully distinguish immediate-FD MILD from SEVERE,
   and the local TD residual stays tiny relative to GAE advantage in the median. **Critic / credit
   locality remains an unresolved mechanism question.**
6. Non-FD gradient interference exists and is more visible after FD100 acquisition, but the
   evidence does **not** establish it as the sole or primary cause of collapse.
7. **No additional tuning run is approved.** In particular, no approval is inferred for
   greater-than-100%-equivalent oversampling or replay, stratified loss weighting, batch-size
   changes, lambda / learning-rate / entropy changes, clipping changes, reward shaping, critic
   architecture changes or observation changes.

### 11.9 Unresolved hypotheses and next research action

The next action is a **fresh read-only mechanism audit** in a new orchestrator chat, before any
further implementation or scientific execution. Priority questions — **an investigation list, not
a conclusion that any one item is defective**:

1. **Actor private observation identifiability:** does the acting ego's immediate-FD private graph
   contain an informative, non-saturated signal distinguishing MILD and SEVERE? Inspect fuel
   normalization, distance / reachability clipping and any other relevant actor-visible features.
2. **Critic conditioning:** why does `value_old` barely distinguish severities despite very
   different targets? Inspect the central observation contents and whether value prediction is
   appropriately conditioned on the current decision / acting ego.
3. **PPO mechanics:** inspect clipping, normalized advantages, repeated PPO epochs and Adam /
   gradient clipping for mechanisms that could acquire and then erase separation.
4. **Gradient interaction:** why do non-FD components sometimes oppose a healthy FD component after
   acquisition?

No new run or code change is authorized by this record.

### 11.10 Limitations and non-claims

- **Two development diagnostic runs, one seed stream, 150 updates each**; the same 20 frozen
  development worlds are re-measured every round (not independent samples); no variance across
  training seeds is estimated.
- The Run A versus Run B comparison changes one configuration key but is **not bit-identical
  physical execution** (BLADE timing nondeterminism); it is not a randomized control.
- Separation pressure is a **local first-order raw-gradient** quantity at PPO epoch 0 only; it does
  not predict the actual Adam step, and epochs 1–3, clipping and gradient-norm clipping are not
  decomposed.
- Credit quantities are descriptive, pooled or matched-within-update associations, **never action
  values**; the TD ratio is not a causal credit decomposition.
- **No causal claim** that FD exposure, non-FD interference, critic conditioning or PPO mechanics
  is the cause of acquisition or collapse.
- Run B's update-75 separation (`+0.0428`, 0 / 20 switches) is a probability-level effect only; no
  selected-action severity conditioning was observed in either run.
- The measured SHA was an unmerged candidate; the semantic-action CTDE R1 (evidence PR #73) remains
  unreviewed and is not used here.
- No confirmatory evidence exists; the confirmatory profile is untouched.

## 12. GENERALIZED-V2 role-only acting-ego CTDE development diagnostic

Recorded on 2026-09-19. The decisions taken on this record are the 2026-09-19 rows of
[`decisions.md` §1](decisions.md#1-decision-log). §11 stays as the record it was; this record
describes measured SHA `68055e39768d5fa601e5960a9f08823b9e65c08f` only.

### 12.1 Scope and epistemic status

| Layer | What it is | Status |
|---|---|---|
| measurement validity | the run executed its recorded authorized plan, completed and reconciled | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** — GPT orchestrator review of 2026-09-19 of the compact evidence package, as transferred in the user-approved packet (§12.2–§12.3) |
| observed findings | critic / credit, held-out behaviour and actor-gradient quantities | reproduced from the original artifacts by the package's extractor (§12.4–§12.6) |
| development interpretation | what the findings are consistent with | **development only**; no causal attribution (§12.7) |
| confirmatory evidence | — | **none**. The confirmatory profile was not selected, inspected or executed |

**Cross-version, not a one-key intervention.** The intervention is a code change — the central
critic state marks the current decision owner's live node with the encoder's existing EGO role
([policy and CTDE §4](../contracts/policy_ctde.md#4-phase-b-ctde)); the comparator is §11's Run B
(FD100) measured at `6ed964a1abd09de2130aee3d0d314c8f32165056`. Comparisons use the common range
through update 100. Every evaluation round re-measures the same 20 frozen development worlds.

### 12.2 Identity and evidence provenance

| Item | Value |
|---|---|
| Run id | `graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` |
| Local original (authoritative, external, untouched) | `C:\gruns\graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` |
| Measured code SHA | `68055e39768d5fa601e5960a9f08823b9e65c08f`, branch `task/ctde-acting-ego-conditioning` — the GPT-approved implementation head of PR #75, **unmerged when measured**; `run_config.json:/provenance/git` records `dirty = false`, 0 dirty paths |
| Comparator | `graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` at `6ed964a1abd09de2130aee3d0d314c8f32165056` (§11, Run B); its original bytes equal the PR #74 index |
| Wall clock (local, 2026-09-19) | 13:34:18 – 14:09:27 |
| Design | GENERALIZED-V2, `ctde`, `p1_milp_v1`, `semantic_k_plus_2_logmeanexp_v1`, DEVELOPMENT profile of manifest `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8` (file SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`), base seed `3000000`, `fuel_damage_mode = seeded_variable`, `fuel_damage_probability = 1.0`, `--actor-gradient-diagnostics` on |
| Budget | 100 updates × 8 successful episodes; at most 12 attempts per update; `eval_every = checkpoint_every = 25`; early stopping disabled |
| PPO / CTDE | unchanged from §11: `gamma = 1.0`, `lr = 0.0003`, `clip_ratio = 0.2`, `entropy_coeff = 0.01`, `n_epochs = 4`, `max_grad_norm = 0.5`; `critic_lr = 0.0003`, `value_coeff = 0.5`, `gae_lambda = 0.95` |
| Configuration boundary | the resolved `train_config` equals the comparator's except `n_iterations 150 → 100` and `output_dir`; every other differing `run_config.json` path derives from `n_iterations` or is provenance |
| Authorized plan / preflight / run config SHA-256 | `ea6752304682a4300b3a5ffd50819a566d6da239fbf7de20e68b9356175f7a19` / `ac51eaef83b631e5c7ac75d54d7163965f0d6a73c29711ee7b1ec0dbb2595705` / `a982aea152192c0f1d2fb8cf96058362d0d8288a3bd5677e8440b07a4f008b43` |
| Compact evidence package | `research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/` on PR #75: byte-identical copies of the small identity, configuration, summary and record files, per-row derived projections of both runs, a two-run `artifact_manifest.json` and a standard-library extractor (`--check`, `--from-derived`). The credit stream, `episode_outcomes.jsonl`, console log, checkpoints, scenarios and plots are identified by SHA-256 only |

### 12.3 Validity facts

| Fact | Value |
|---|---|
| Productive updates | 100 / 100 |
| Training episodes successful | 800 / 800 over 800 attempts (budget 1200); 0 failures, 0 replacements |
| Trained transitions (`run_summary` = Σ `train_records.n_transitions` = credit rows) | 2982 |
| Training clean / damaged successful; FD events / immediate-FD wakes | 0 / 800; 800 / 800 |
| MILD / SEVERE training episodes | 421 / 379 |
| Evaluation rounds; episodes successful | 5; 300 / 300 |
| Every round: base cells defined / groups metric-eligible | 10 / 10; 20 / 20 |
| Held-out check | overlap 0 against `[3000000, 3001200)`, entire manifest |
| Gradient diagnostic rows; defined contrast | 100, one per update; 99 / 100 |
| `accounting_reconciled`; `native_exit_code.txt`; Traceback lines in console | `true`; `0`; 0 |

### 12.4 Critic / credit findings

Immediate-FD rows of the FD-selected ego (800 per run over iterations 0–99). Within-update
`SEVERE − MILD` is `mean(severe) − mean(mild)` per update holding both severities, median over
those updates (99 over 0–99; 50 over 50–99).

| Quantity | Acting-ego 0–99 | 50–99 | Historical FD100 0–99 | 50–99 |
|---|---|---|---|---|
| `value_old` | **`-0.02049`** | **`-0.02362`** | `-0.0000279` | `-0.00000186` |
| `value_target` | `-0.330` | `-0.369` | `-0.317` | `-0.361` |
| `raw_advantage` | `-0.319` | `-0.357` | `-0.316` | `-0.360` |
| `td_residual` | `-0.00178` | `-0.00306` | `-0.0000886` | `-0.000109` |

`value_old` by 25-update window (median): acting-ego `-0.0024` / `-0.0549` / `-0.0132` /
`-0.0583`; historical `-0.0006` / `+0.0001` / `-0.00003` / `+0.000007`.

Nonterminal immediate-FD decisions (not the episode's last decision; acting-ego 706, historical
699; local = `td_residual`, future = `raw_advantage − td_residual = γλ·A_{t+1}`):

| Quantity | Acting-ego | Historical FD100 |
|---|---|---|
| median `|td_residual| / |raw_advantage|` | **`0.047006`** | `0.0030719` |
| mean `|local|` / mean `|future|` | `0.123` | `0.0547` |
| fraction `|future| > |local|` | **`0.92068`** | `0.97997` |
| within-update `SEVERE − MILD` future, median (0–99) | `-0.311` | `-0.319` |
| within-update `SEVERE − MILD` local, median (0–99) | `-0.00061` | `+0.000077` |

### 12.5 Behavioural findings

Held-out DEVELOPMENT macro `SEVERE − MILD P(SELF_PRESERVATION_ABORT)` at the certified ego's
immediate-FD wake; directional switches out of 20 matched groups, reverse switches 0 throughout.

| Update | Acting-ego | Switches | Historical FD100 | Switches |
|---|---|---|---|---|
| 0 | `-0.000516` | 0 / 20 | `-0.000516` | 0 / 20 |
| 25 | `-0.0000357` | 0 / 20 | `+0.0000907` | 0 / 20 |
| 50 | `+0.0000326` | 0 / 20 | `+0.00208` | 0 / 20 |
| 75 | `+0.007641` | 0 / 20 | `+0.042798` | 0 / 20 |
| **100** (primary endpoint) | **`-0.001091`** | 0 / 20 | `-0.00000865` | 0 / 20 |

### 12.6 Actor-gradient context

Updates with a defined contrast (25 / 25 in each window); counts are positive-pressure updates.

| Window | Run | FD + | non-FD + | total + | FD + ∧ total − | FD median | FD mean |
|---|---|---|---|---|---|---|---|
| 50–74 | acting-ego | 15 / 25 | 15 | 15 | 2 | `+0.00213` | `-0.00545` |
| 50–74 | historical | 20 / 25 | 21 | 18 | 3 | `+0.0195` | `+0.00683` |
| 75–99 | acting-ego | 18 / 25 | 11 | 13 | 6 | `+0.00122` | `-0.00023` |
| 75–99 | historical | 13 / 25 | 9 | 12 | 2 | `+0.00026` | `-0.0128` |

The entropy term flips the sign of separation pressure in 1 / 99 defined updates (historical
0 / 99 over the same range); the minimum surrogate-versus-actor-loss cosine is `0.878`.

### 12.7 Development interpretation

- **Role-only acting-ego conditioning materially improved critic / credit localization:**
  `value_old` now separates severities at the immediate-FD decision (median `-0.02049` versus
  `-0.0000279`), and the local TD share of the advantage rose about fifteen-fold (`0.047006`
  versus `0.0030719`). **The absence of acting-ego context was a real critic-conditioning
  defect.**
- **That mechanism improvement did not improve held-out acquisition or retention:** the update-75
  macro is lower than the historical FD100 run (`+0.007641` versus `+0.042798`), the update-100
  macro is effectively zero (`-0.001091`), and directional switches stay `0 / 20` at every
  evaluation point. **Role-only conditioning is insufficient to solve policy acquisition /
  retention.**
- **Severity-specific GAE credit is still mostly future-dominated** (`|future| > |local|` in
  `92.1 %` of nonterminal immediate-FD decisions; the within-update severity gap sits almost
  entirely in the future term).
- **Actor-gradient context:** the role-only intervention does not produce stronger pre-collapse FD
  pressure than historical FD100 (50–74: `15 / 25` versus `20 / 25` positive; 75–99: `18 / 25`
  versus `13 / 25`). This does **not** establish a simple monotonic relationship between improved
  locality and separation pressure.
- **No claim** that critic conditioning is the sole cause of collapse.

### 12.8 Next hypothesis

The value head still reads the acting ego only through a mean pool over all node embeddings, so
the decision owner's signal may be diluted. The next minimal mechanism test is an **explicit
acting-ego readout**: `V = ValueHead([global mean pool ; acting-ego post-message-passing
embedding])`, adding no new information — a dedicated readout channel only
([`decisions.md` §1](decisions.md#1-decision-log), 2026-09-19). Its implementation needs its own
exact-candidate review and any run its own authorized bounded plan.

### 12.9 Limitations and non-claims

- **One development diagnostic run, one seed stream, 100 updates**; no variance across training
  seeds is estimated; the same 20 frozen worlds are re-measured each round.
- **Cross-version** against a historical run (different measured code); not bit-identical
  physical execution (BLADE timing nondeterminism); not a randomized control.
- Credit quantities are descriptive associations, **never action values**; separation pressure is
  a local first-order epoch-0 quantity.
- **No confirmatory claim**; the confirmatory profile is untouched. The measured SHA was an
  unmerged candidate.
