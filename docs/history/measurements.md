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
| GENERALIZED-V2 benchmark preflight (produced the external manifest) | **unverified** — no preflight evidence is in the repository, and the consuming runs' measured SHA does not establish the producer's | external manifest, see §7; not part of either evidence package | no authorization or review record in the repository |
| GENERALIZED-V2 development R1 — actor-only | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #61, head `1375a881637a9a32721a1630f598adc571422a47` | no verdict recorded in accessible artifacts |
| GENERALIZED-V2 development R1 — CTDE | `ae42cb01677f94868b2873008d87be677e31f0c8` | evidence PR #62, head `b2bbe7a6235c3b9255106826cfb268af7e73f72d` | PR #62 records a prior GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT`; not independently verified |

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
  repository.
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
