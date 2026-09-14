# Implementation history

> **Historical record — not instructions.** It preserves every implementation and lock entry of
> the former `CLAUDE.md` §7, closure narratives from the former `CLAUDE.md` §8, and a ledger of
> merged pull requests. Entries were moved **verbatim** from base
> `ae42cb01677f94868b2873008d87be677e31f0c8`. A statement inside an entry that something is
> "next", "open", "not authorized" or "the sole writable task" was true on its own date only.
> Inside moved text a bare `§N` means that former `CLAUDE.md` section
> ([compatibility index](../../CLAUDE.md#8-compatibility-index-for-older-references)). Current
> contracts are in [`docs/contracts/`](../contracts/); measurement entries are in
> [`measurements.md`](measurements.md).

## 1. The hash convention

> **Hash convention:** a commit cannot cite its own SHA, so each lock's hash is recorded in the NEXT commit that touches this file — never in the lock itself. The amend route (commit → fill the hash → `--amend`) is **DEPRECATED**: amending shifts HEAD to a new SHA, leaving the recorded hash pointing at the dangling pre-amend commit (that is what produced the stale `f831e69`, fixed here to `95c3189`).

As a live procedure this convention is replaced by
[`cc_review.md` §5](../workflows/cc_review.md#5-code-and-documentation-together): documentation
records integration SHAs only when materially needed, and never its own. It remains the reason
each lock's integration SHA appears in a later entry below.

## 2. Implementation and lock entries

- `777bd85` — executor per-ego private task lists (no-comms ISO-1..3).
- `9a211ab` — encoder `task_feat_dim` derives from builder `TASK_FEATURE_DIM` (fix 5/6 desync).
- `1ba583c` — Stage-0 episode setup + `Belief` (allocated-only baseline, independent per-ego beliefs).
- `b4a29ba` — two-phase tick-loop + `Policy`/`Transition` seam.
- `87cb17b` — terminal utility-regret reward.
- `ae97d7d` — real discovery-chain split on unified `DETECTION_KM` (generator + split + executor).
- `95c3189` — full-pipeline rollout harness (pure consumer; 20-ep validation).
- `f680710` — BLADE playback recording wired (armed in setup via `EpisodeContext.record`, driven in `run_episode`; purity-proven, TEST 1b).
- `561b7cb` — docs: §7 hash-convention inline + recording-lock SHA fill.
- `814734e` — graph_builder: inline `_compute_fuel_norm` (sever the flat observation seam; Step 1 of the flat-path cleanup).
- `d9b8c17` — strip the five flat `__init__` re-export vectors + lock `tests/test_import_purity.py` (Step 2 of the flat-path cleanup).
- `ab54ac3` — delete the retired flat path: 38 paths removed (Step 3 of the flat-path cleanup).
- `7f324fd` — doc hygiene + workspace pruning: **Step 4, the FINAL lock of the cleanup phase.** ~20 stale docstring/comment sites reworded across the nine graph modules + `blade_executor_minimal` + `graph_executor_smoke` (present-tense references to the deleted flat path → truthful today or past-tense provenance); README's two stale `train_full.py` sites fixed; `.gitignore` gains `generated_scenarios/`; workspace pruned to two worktrees (main + `../flat-baseline`) with the three stale `claude/*` branches deleted. Comment-only in code — zero code lines changed; all six layer selftests + 12/12 import-purity green.
- `b96d29f` — **final doc sweep** (post-cleanup coda). Four flat-era docs deleted: `LOGS_GUIDE.md`, `RUN_SUMMARY.md`, `docs/INTEGRATION_GUIDE.md`, `docs/MATCH_AOU_API.md` — the first three document deleted code; the fourth documents the live solver but through a dead API (`StepType`, removed in `5eeaf3c`) in every example. All four preserved on `flat-final` (`4d44c34`) + the annotated tag `pre-cleanup` (peel it: `pre-cleanup^{commit}` → `561b7cb`). README's Documentation section lost its three dead links (the two deleted `docs/` files + `RL_MODULE_DOCUMENTATION.md`, orphaned back in `ab54ac3`), leaving only the live `BLADE_API_DOCUMENTATION.md`. Untracked mid-cleanup snapshot `src/match_aou.zip` deleted. `training_output*/` added to `.git/info/exclude` — a local-only, never-tracked safety net shared by both worktrees, deliberately broader than `.gitignore`'s `training_output_*/`. Docs-only; 12/12 import-purity green.
- `be3729d` — per-episode RNG reseed in `graph_rollout` (PPO-phase step 1).
  Every episode now reseeds global `random` + torch with `base_seed+i` at the top
  of its iteration (the generator already uses its own `random.Random(seed)`),
  making episode i a pure function of its seed given the policy weights (still
  pinned ONCE before the loop by `torch.manual_seed(base_seed)`). Records gain
  `known_target_ids` — the t=0 known-split identity, snapshotted from
  `ctx.beliefs` BEFORE `run_episode` (after wakes the N beliefs legitimately
  diverge per ego; that divergence is the no-comms guarantee). Proven by a
  throwaway two-part check: (T1) same-config double run ⇒ field-identical
  records; (T2) episode-in-isolation reproduces the split of the same seed
  inside a longer run, with scenario content per seed differing ONLY in the two
  unseeded episode-tag fields (`/currentScenario/id` uuid4 + `/name` episode
  index — a finding to remember: scenario ids are NOT seed-derived; unit ids
  ARE template-stable). Regression: 15/15 pytest incl. 12/12 import purity.
- `830bd32` — `evaluate_action` + shared `_masked_dist` construction site in
  `graph_action` (PPO-phase step 2). `sample_action` and `evaluate_action` now
  build the joint masked distribution through ONE private helper, so rollout and
  PPO-update distributions are identical BY CONSTRUCTION (drift there is a silent
  research bug). `evaluate_action(logits, mask, meta, node_v) -> (log_prob,
  entropy)`: grad mode is caller-controlled (no no_grad inside); masked or
  out-of-bounds stored cells raise ValueError (fail loud — a masked stored action
  means mask reconstruction diverged). Proven: bit-identical pre/post-refactor
  baseline incl. a 50-draw RNG sweep; bitwise sample/evaluate agreement; epoch-0
  ratio exactly 1.0 on a real policy; finite grads on every exercised
  encoder+head param (edge_attr_proj legitimately unexercised, exact-name
  whitelisted); masked/out-of-bounds guards. Regression: 20/20 pytest incl.
  12/12 import purity; tick-loop selftest green end-to-end.
- `628e45f` — `graph_ppo`: the PPO core, Phase A actor-only (PPO-phase step 3).
  EpisodeRecord (per-ego chains — the Phase-B GAE seam contract) + PPOBuffer +
  compute_returns_and_advantages (THE REPLACEABLE COMPONENT: return == episode R at
  the dormant gamma=1.0; baseline = mean R over EPISODES incl. zero-wake;
  advantages normalized with eps guard) + clipped_surrogate + PPOUpdater (one Adam
  over encoder+head, per-transition re-encode -> rebuilt mask -> evaluate_action ->
  clip, entropy bonus, one backward/epoch, grad-norm clip; empty batch = clean
  no-op; NO value loss — PHASE-B SEAM comments mark where the critic joins).
  *(That is the Phase-A actor-only state this commit built, and it is still exactly what
  `PPOUpdater` does. The critic joined LATER, as a SEPARATE `CTDEUpdater` beside it —
  see the Phase-B CTDE lock at the end of §7.)*
  Proven in _selftest + tests/test_graph_ppo.py (18 tests): epoch-0 ratio == 1 and
  loss == -mean(A_norm); learning direction (positive-advantage action rises);
  clip branches hand-checked + clip_fraction > 0 live; per-ego grouping order;
  degenerate batches (all-same-R, empty, all-zero-wake) NaN-free; finite grads
  (edge_attr_proj exact-name exempt); import purity green.
- `21e4d14` — `graph_train`: the outer PPO Trainer, Phase A (PPO-phase step 4 — the
  LAST piece of Phase A). New leaf module `rl/training/graph_train.py`, purely
  additive: no locked layer touched, and deliberately NOT in the import-purity
  ENTRY_MODULES (it imports BLADE, like graph_rollout). Wraps the locked pipeline into
  a real run: per iteration it collects `episodes_per_iteration` stochastic episodes
  through the rollout skeleton (one generator, per-episode reseed, env.close in
  finally) into a fresh PPOBuffer, runs ONE PPOUpdater.update built ONCE for the run,
  clears, and appends a scalar record. Owns the seeding schedule: train seed =
  base_seed + (iteration*eps + j); eval on a FIXED DISJOINT band (eval_base_seed + e),
  enforced by TrainConfig.validate (overlap raises). Deterministic eval every N iters
  on the held-out band (no buffer / no update). Save-only checkpoints
  (encoder+head+optimizer+PPOConfig); resume DEFERRED. 3-panel plot (learning curve vs
  R=0 oracle ceiling / meta-action mix / entropy) from the jsonl, drawn in a CHILD
  process with KMP_DUPLICATE_LIB_OK — torch+matplotlib abort together on this
  Windows/OpenMP stack, so the flag is confined to a numerics-free child and training
  never depends on matplotlib. Proven: tests/test_graph_train.py (8 tests — checkpoint
  round-trip incl. optimizer state, seed-schedule + band disjointness, plot-from-jsonl;
  suite 38 -> 46, import purity 12/12) and _selftest under nlp_env (short real run;
  EVAL PURITY — train records byte-identical eval-on vs eval-off; honest zero-wake via
  max_ticks=5). FINDING (see §8): the Trainer is correct, but every episode returns
  R ~ -1/3, so a baseline run as-is will NOT learn — a reward/scenario issue, not a
  Trainer bug.
- `95c09dd` — **Phase-A baseline scenario cell + config visibility** (`graph_train`
  only; no locked layer touched). Defaults retargeted from the measured-degenerate
  `(3,3)` to the SELECTED cell: `num_red_airbases=(6,6)`, `partial_ratio=0.5` →
  **known 3 / hidden 3**, `U_oracle=480`. Why this cell: 6 targets > the 4-agent fleet
  (which comes from the base template `strike_training_4v5.json`, NOT a config knob)
  removes the forced 2:1 stacking that pinned every episode at R = −1/3; `known ≥ 3`
  keeps bonmin out of its B&B symmetry stall (~15 min/episode at `known ≤ 2`);
  measured `std(R) = 0.1443`, split outcome clean 11/12, zero dominating-set breaks.
  New in the module: **`derived_split`** — a MIRROR of `split_tasks`'
  `max(1, int(n·partial_ratio))` and the single arithmetic site behind the startup
  echo, `validate()`'s hazard warnings, and `run_config.json`; its equivalence to the
  locked authority is test-enforced over an n × ratio grid that INCLUDES the degenerate
  `n < 2` branch, asserted against real `split_tasks` returns rather than only its
  `meta`. `TrainConfig.split_preview` previews both ends of a range. CLI gains
  `--num-red-airbases` (`N` or `LO,HI`), `--partial-ratio`, `--stretch-target-ratio`,
  every default read OFF `TrainConfig` (drift-guarded by test); `include_sams` and any
  radius are deliberately NOT exposed. `validate()` now WARNS — never raises — on
  `known < 3` and `hidden == 0`, judged at the range's LOW end (both quantities are
  non-decreasing in n, so that is the worst case). Every run writes
  `run_dir/run_config.json` (full resolved config incl. nested `PPOConfig`, the derived
  split, the base-scenario name) and the startup header echoes `known/hidden` — the
  standing defence against the truncation trap: `int()` truncates, so at n=6
  `1.0/3.0` gives known 2 while the decimal `0.333` gives known 1. Never
  auto-corrected: the config you type is the config you get, and the header tells you
  what it is. Proven: suite 46 → 59, import purity 12/12, `--selftest` green
  end-to-end under `nlp_env`. LIVE EVIDENCE the §8 blocker is lifted:
  `adv_std_raw = 0.1197` (was ~0), per-episode R spanning −0.5 … −0.1667 (exact
  multiples of 80/480), three distinct iteration means, and `OPPORTUNISTIC_ENGAGEMENT`
  firing 7× in 12 episodes — matching the instrument's measurement for this cell
  exactly.
- `384845b` — **Scenario-construction preconditions** (step 1 of 3 of the offline
  scenario-construction phase; no locked layer touched). Three fixes the inverted
  build order depends on. (1) **LAUNCH POINT.** The base template parked the four
  BLUE aircraft at `(32.35416…, 34.81240…)` while their own airbase sits at
  `(32.85416…, 35.31240…)` — 72.7 km away — and `Game.launch_aircraft_from_airbase`
  only moves the object between lists without repositioning it, so every episode
  put the fleet airborne 72.7 km from its base. The four aircraft records now carry
  the airbase's coordinates. The template JSON is **MINIFIED** (one line, no
  newlines): edit it by exact string replacement and NEVER round-trip it through
  `json.dump`. Intended consequence: `Agent.location == Agent.return_location`, so
  the solver's `round_trip_cost` is now a symmetric out-and-back instead of a
  launch→target→base triangle, and a given seed MAY now yield a different
  allocation. (2) **The source of that skew** — `_adjust_aircraft_count`'s
  empty-inventory branch placed a new aircraft at `base − 0.5°/0.5°`; it now anchors
  to the base, making the defect unreproducible. (3) **`VariationConfig.
  ensure_discovery_chain: bool = True`** gates Layer 1's CALL SITE (body untouched).
  Default = today's behaviour; `False` skips the relocation pass, which the
  construction path requires: with only the KNOWN targets generated, Layer 1 would
  cluster them into ≤`DETECTION_KM` pairs and collapse the route diversity that
  hidden-target placement is measured against. When `False`, the seven stats keys
  Layer 1 stamps (`easy_relocated`/`_total`/`_isolated`, the three `stretch_*`,
  `min_radar_km`) are ABSENT from `last_generation_stats` — read them with `.get`,
  never `[...]` (`graph_episode_setup._selftest_generator` indexes `min_radar_km`
  directly and is safe only because it runs with the chain ON). Verified: no seeded
  rng consumer runs after Step 5.25, so gating the call does not shift the rng
  stream. Proven: `tests/test_scenario_construction_preconditions.py`, suite 59 →
  64, import purity 12/12, both module selftests + the bonmin selftest green under
  `nlp_env`. The load-bearing test is **P6** — the four RED-airbase coordinates for
  a fixed seed are byte-identical before and after, which makes the phase's
  foundational claim falsifiable: target placement is a function of the BASE
  coordinates and the rng stream ONLY, never of the aircraft's own position. **P5**
  proves the switch is a true skip by monkeypatching `_ensure_discovery_chain` to
  raise (`generate()` has no try/except, so the raise cannot be swallowed).
- `a5a4137` — **workflow + handoff migration** (documents only; no code touched, no test
  delta). §1 moves from STOP-before-commit / local-only to Git transport: an explicit base
  SHA, a reviewable candidate commit, and a mandatory status block. Adds the **grade = trust
  policy** definition that §1 previously referenced without defining. Fills the two SHAs the
  hash convention deferred (`95c09dd`, `384845b`; the entries above carried `PENDING` until
  this commit). The continuing handoff (`graph_rl_project_handoff.md`) lands in the SAME
  commit: it declares each task's grade, so the two documents are only coherent together.
  **Corrected in place (docs-only, no separate history entry):** that entry asserted that
  both orchestrators read this repository through equivalent direct Git connectors, and that
  a task branch + draft PR is the one universal transport. Both claims are false. Access is
  **capability-aware**, and §1 now carries two transport modes over ONE shared `CLAUDE.md`
  and ONE shared handoff — no per-orchestrator forks of either document. `GPT_GITHUB`: the
  GPT orchestrator resolves branches, PRs, files, and exact SHAs through GitHub and reviews
  the exact `base...candidate` comparison. `CLAUDE_MOUNTED_MAIN`: the Claude orchestrator's
  shared view is CC's synchronized mounted checkout of `main`, so the reviewable artifact is
  an exact post-push `main` SHA plus focused hunks and targeted test evidence requested from
  CC — task branches and PRs are not assumed reachable. The packet or the user declares the
  mode; it is never inferred.
- `f319095` — **shared-document workflow correction** (documents only; no code, no test
  delta). §1's `CLAUDE_MOUNTED_MAIN` description and handoff §0 now state that side's real
  capability: a synchronized mounted snapshot of `main` as a **search** interface that cannot
  diff, cannot prove absence, and can lag. Adds the **Grade-A routing default** — Grade A goes
  to `GPT_GITHUB` when available because it is the only mode that gates `main` behind a
  branch; Grade A under `CLAUDE_MOUNTED_MAIN` is a declared exception with mandatory hunks +
  targeted evidence from CC, and no lock or dependent work before exact-SHA approval. The
  grade-definition and output-discipline sites became pointers to that one bullet rather than
  second copies.
- `d6758ac` — **B1: offline scenario-construction configuration — CLOSED / MERGED /
  LOCKED** (known-only cell, step 1 of 3 of the offline scenario-construction phase;
  integrated into `main` by merge commit `bd087c3`, PR #2). States the reference cell
  outright instead of deriving known/hidden from a ratio: `TrainConfig` gains
  `num_agents=3` (`<= n_known`), `n_known=3`, `n_hidden=3` (PLANNED for B2/B3 — B1 emits
  ZERO hidden targets), `min_target_distance_km=200.0`, `min_known_separation_km=100.0`.
  `build_variation_config` is the ONE site turning a `TrainConfig` into the generator's
  `VariationConfig`: exactly `n_known` targets, `ensure_discovery_chain=False` (Layer 1
  disabled ONLY on this construction path — it would cluster known targets and flatten
  route diversity), `strict_geometry=True` (the generator raises rather than silently
  weakening the requested geometry). The review fix re-measures the ring-sampled
  candidate with the real `_haversine_km` — the ring's flat-earth degree-conversion had
  let true sub-floor targets through a 300-seed sweep — so `strict_geometry` now enforces
  a TRUE great-circle `min_target_distance_km` / `min_target_separation_km` floor;
  STRICT-only, so every legacy non-strict caller (incl. `P6`'s pinned fixture) is
  byte-unchanged (P9c, P11). `RolloutConfig` (`graph_rollout.py`) mirrors the same
  reference-cell fields field-for-field and now validates them as `run_rollout`'s FIRST
  statement, before any directory, policy, generator, or BLADE import — closing the
  `TrainConfig`/`RolloutConfig` divergence recorded in §8. Does NOT build hidden-target
  placement (B2/B3 work). Proven: suite 64 → 84, import purity 12/12, module selftests
  and the bonmin selftest green under `nlp_env`.
- `e22aee3` — **B2: route-relative hidden-target placement — CLOSED / MERGED /
  LOCKED** (step 2 of 3 of the offline scenario-construction phase). Reviewed code SHA
  `e22aee359e06591bdb179ef06a566db90f83a558`, integrated into `main` by merge commit
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3). New leaf module
  `rl/training/graph_hidden_placement.py` + `tests/test_graph_hidden_placement.py`; no
  existing file touched, so no locked layer moved. **PURE**: no BLADE, gym/gymnasium,
  torch, solver, `setup_episode`, file I/O, or module-global randomness — `rng` is an
  explicit required `random.Random`, and `detection_km` arrives through
  `PlacementParameters` rather than being imported from `graph_episode_setup` (importing
  it would drag the layer into the setup/solver/executor closure). Contract:
  `place_hidden_targets(solution, belief_tasks, launch_point, parameters, rng) ->
  Tuple[HiddenPlacement, ...]`, egos iterated in SORTED id order so the result never
  depends on the solution dict's insertion order; reproducibility is judged by
  `geometric_fingerprint` (coordinates only — **no UUIDs**, per §8's "Added enemy airbases
  are not seed-stable by id"). **Route prediction reuses the SHARED
  `nearest_neighbor_order`** (imported, never reimplemented — originally from the minimal
  executor, relocated to `utils/scheduling_utils.py` by `2a3f89c` with the body unchanged):
  ascending `level_order`, the
  helper called separately inside each level, its returned end location chained into the
  next, first level seeded from the shared launch point — so prediction cannot drift from
  execution. **Cardinality: exactly ONE placement per non-empty ego route**; a general
  `n_hidden != usable ego routes` distribution is a separate future design task and is NOT
  solved here. **Geometry:** only `G = L - D` of a leg is guaranteed flown (inside `D` of
  the target the ego attacks and issues no new movement); the perpendicular PROJECTION sits
  at `s = f·G` with `f ~ Uniform[0.60, 0.85]`; sensing guard `10 km`; leg-1 max |offset| =
  `D - guard` (40 km at `D = 50`); a later leg budgets residual origin uncertainty
  `(1 - s/L)·D` and caps |offset| at `D - guard - origin_uncertainty`, and its whole
  approved fraction interval must project beyond the uncertain origin vicinity. **Later
  legs require the STRICT nearest-neighbor condition `gap > 2·D`** (equality rejected; one
  remaining candidate passes trivially). Selection: uniform among eligible later legs,
  else fall back to a valid leg 1, else raise. Everything fails LOUDLY —
  `HiddenPlacementError`, no silent clamping and no weakened margin — and every returned
  placement is re-measured by an INDEPENDENT bearing-based cross-track/along-track path
  before it is returned. Two review fixes are part of the locked behaviour: **F1** —
  `_as_assignment` never coerces; fields must be genuine `numbers.Integral` values (a numpy
  integer still works, normalized to `int`), `bool` is rejected despite subclassing `int`,
  and fractional floats, integral-VALUED floats and numeric strings all raise (`int(...)`
  had silently accepted `(0.9, 0, 0)` AS `(0, 0, 0)`, quietly changing the predicted
  route). **F2** — `validate_placement` checks the recorded `tie_margin_required_km` for
  EVERY `leg_index > 1` BEFORE branching, so the `single_candidate` path can no longer skip
  the requirement; missing, non-finite and incorrect values all raise. Verified on the
  integrated merge: 18 focused B2 tests, 12/12 import purity, full suite 100 → **102**,
  `git diff --check` clean, plus all 18 B2 tests and all 12 import-purity entry modules
  green through the `nlp_env` `__main__` runners. No bonmin or live BLADE run is involved.
  Consumed by construction-mode `setup_episode` from B3 (`dd14ab4`) onward.
- `dd14ab4` — **B3: the setup seam — CLOSED / MERGED / LOCKED** (step 3 of 3 of the
  offline scenario-construction phase; this closes the phase). Reviewed code SHA
  `dd14ab418c71e3bd615f1198d0c612502642d29b`, integrated into `main` by merge commit
  `14224531db9deb700f6e397203177eb8c701c6cc` (PR #4); the merged tree is byte-identical
  to the approved one (`git diff --quiet dd14ab4 1422453`). `setup_episode` gains TWO
  EXPLICIT PATHS chosen by the `(n_hidden, placement_rng)` PAIR — both omitted keeps the
  unchanged legacy `split_tasks` path, both supplied runs CONSTRUCTION, exactly one
  raises before any BLADE object exists, and the mode is never inferred from
  `partial_ratio`. Construction is **solve → place → patch → reload**: a known-only env-1
  is solved for `A_init`, the LOCKED B2 layer places one hidden target per non-empty ego
  route, `build_patched_scenario` appends exactly that many enemy airbases to
  `currentScenario.airbases` (deep-copied prototype, fresh uuid4, deterministic name,
  empty inventory, known targets and their positions untouched), env-1 is closed, and
  env-2 is reloaded on the patched JSON. **Env-2 is the sole runtime source of truth**:
  agents and tasks are re-extracted from it, agent IDs must survive the reload as an
  ORDERED list, known belief tasks are re-materialized from env-2 BY TARGET ID in A_init's
  positional order (so `task_idx` stays valid), the oracle is an independent solve over
  ALL env-2 targets, and no env-1 `Agent`/`Task` object reaches the returned context.
  Guards fail LOUDLY and never repair: airbase-only cell (`TrainConfig` /
  `RolloutConfig` `validate()` also reject `include_sams=True`), shared launch point plus
  `Agent.location == Agent.return_location`, exact `len(placements) == n_hidden` (no
  truncation, padding, duplication or redistribution), unsafe/ambiguous prototype, name/id
  collision, agent-id drift, known-target loss, world cardinality. `n_hidden=0` is a legal
  probe that never calls `split_tasks`. `EpisodeContext.placements` is the id-free
  placement audit and construction `split_meta` reports truthful
  `known/hidden/partial/full` plus a coordinates-only `geometric_fingerprint`;
  `graph_train` / `graph_rollout` now pass `n_hidden` + a fresh `random.Random(seed)` and
  report REAL emitted counts (`n_targets_emitted == n_known + n_hidden`). **Private-sensing
  isolation is proven through the integrated setup/tick seam** (not by re-testing
  `Belief.independent`): a setup-constructed hidden world target reported as sensed only by
  ego A enters ONLY ego A's belief and executor slice through the unmodified `run_episode`
  Phase-1 chain, while every peer belief and slice stays byte-unchanged and the target was
  in NO belief at t=0. One review fix is part of the lock: **`_build_env` now owns cleanup
  until it returns successfully** — a failure in `env.reset()` (or the side selection after
  it) sits in a window no caller guard can reach, so it closes the environment exactly once
  and re-raises the original exception unchanged; the regression test drives `_build_env`
  directly and was verified to FAIL against the pre-fix body. Verified: base suite
  **118 passed, 4 skipped** (102 → 121 collected), 12/12 import purity, `git diff --check`
  clean; **19/19** `tests/test_graph_setup_seam.py` checks under `nlp_env`, plus the
  placement, train, import-purity and legacy `graph_episode_setup` runners. Live seed-0
  reference rollout: 3 agents, **3 known + 3 hidden = 6 full targets**,
  `U_oracle = 479.99968` (the frozen solver EPSILON form of a raw 480), **4 organic wakes**,
  `ended=done`, reward `-0.3333`, both bonmin solves successful, no `CRASH`/`Traceback`.
- `1b48145` — **B4: auditable training-run instrumentation — CLOSED / MERGED / LOCKED.**
  Reviewed code SHA `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c`, integrated into `main` by
  merge commit `ba936606deada050ed9298600ee9041fc330af6c` (PR #6); the merged tree is
  byte-identical to the approved one (`git diff --quiet 1b48145 ba93660`). **PREPARATION,
  NOT A MEASURED BASELINE — no training run was performed.** Two files only
  (`rl/training/graph_train.py`, `tests/test_graph_train.py`); the full contract is in §5
  ("Trainer + run auditability"). Grade A under `GPT_GITHUB`: the first candidate
  (`dc2142627dc40886667170fc2121fe50336329cd`) was REQUEST-FIXES, and the fix chain landed
  as a NEW commit on the same branch — the reviewed commit was never amended, rebased or
  force-pushed, and `1b48145` is the approved head. **Review fixes, both now part of the
  lock:** (F1) an all-failed iteration was being counted as ZERO-WAKE because both states
  end at `n_epochs_run == 0` — `_iteration_outcome` now classifies `all_failed` /
  `zero_wake` / `productive` as three disjoint states from episode counts, the summary
  carries all three counters, and the console prints at most one flag instead of both;
  (F2) provenance was collected AFTER the run directory existed, so a run's own untracked
  artifacts under an in-repo `output_dir` could register as dirty source state — collection
  now precedes every artifact, `available=True` requires the SHA **and** the clean/dirty
  verdict (a failed `git status` no longer leaves `available=True` with `dirty=None`; the
  recovered SHA is still reported with an explicit reason), and `train` refuses incomplete
  provenance after writing the attempted `run_config.json`. A third requested fix
  (a duplicate `seed = train_seed(...)`) was NOT APPLIED: it is not present at the reviewed
  SHA — the assignment occurs once, and a whole-file consecutive-duplicate scan found only
  the intentional nested `try:` in `_run_one_episode` (the outer owns the env-closing
  `finally`, the inner attributes the `setup` stage). **Incidental measured fix:**
  `bonmin -v` emits byte `0x81`, and `subprocess` `text=True` decodes on a reader THREAD —
  the `UnicodeDecodeError` killed that thread, printed a traceback on every run and returned
  an EMPTY stdout with rc 0, so the probe would have recorded `ok` with no output; probes
  now capture bytes and decode leniently (`_probe_command`). `_stats` was removed as
  orphaned by this change (`graph_rollout` keeps its own separate copy). Verified:
  `tests/test_graph_train.py` **55 passed** (34 → 55), base suite **139 passed, 4 skipped**
  (118 → 139), 12/12 import purity, `git diff --check` clean, and all **55** green through
  the standalone `__main__` runner under `nlp_env`. **No `graph_train --selftest`, no live
  BONMIN training probe and no real training run were executed** — the tests are
  solver-free, driving `train` through stubbed episode/generator/update seams and an
  injected Git verdict.

- `211e12e` — **B4 follow-up: per-episode observability, unique-target semantics and
  per-round eval artifacts — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `211e12e49b676637362d42effdb80988dd0e55eb`, integrated by merge commit
  `ffb95a6ee90df45b2d89802b321dcadcbc272821` (PR #7). Exactly two files changed:
  `src/match_aou/rl/training/graph_train.py` and `tests/test_graph_train.py`; policy,
  reward, PPO, executor, tick-loop, scenario content and seed semantics are unchanged.
  Every successful attempt now prints one immediate `OK` block; unique-target aggregates
  are derived directly from unique target ids and never from display names; structural
  roster defects were routed to accounted `setup` failures instead of false successful
  zeros — **that ROUTING is SUPERSEDED: since `36365f2` such a fault is a
  `MeasurementIntegrityError` and ABORTS the run**, and the false-successful-zero fix this
  PR made is unaffected; and each eval round keeps a disjoint scenario-tag namespace while
  reusing the same held-out seeds. Grade A under `GPT_GITHUB`: candidate
  `24241690572a7a5264e24348db5e9412b41bc47a` received REQUEST-FIXES because a degraded
  roster could silently report a false `0/0` and its docstring claimed the helper never
  raised. The correction landed as a NEW commit, never amend/rebase/force-push, and
  `211e12e` was reviewed and approved. Verified at the approved head:
  `tests/test_graph_train.py` **73 passed**, import purity **12 passed**, full suite
  **157 passed, 4 skipped**, standalone `nlp_env` runner all **73 passed**, and
  `git diff --check` clean. One authorized smoke produced three `OK` blocks, the seed-0
  reference `R=-0.3333` / 4 wakes / `targets_confirmed_unique=4/6`, and coexisting
  pre/post eval scenario files; that smoke validates implementation only and is not a
  scientific result.
- `a8669f4` — **FD-BASELINE-v1: the deterministic, ego-local fuel-damage difficulty —
  CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `a8669f450708c2508753c49ab16fd1028b29607d`, integrated by merge commit
  `1cecb0ac99f839d47ffeea12c8871aec77e66640` (PR #8); the merged tree is byte-identical
  to the approved one (`git diff --quiet a8669f4 1cecb0a`). Grade A under `GPT_GITHUB`.
  The full technical contract is in §5 ("FD-BASELINE-v1 — the difficulty factor") and the
  tick placement in §4; this entry records the LOCK, not the mechanism.
  **Reviewed scope: SEVEN cumulative files** — `rl/training/graph_fuel_damage.py` and
  `tests/test_graph_fuel_damage.py` (both new), plus `rl/action/graph_trigger.py`,
  `rl/training/graph_tick_loop.py`, `rl/training/graph_train.py`,
  `rl/training/graph_rollout.py` and `tests/test_graph_train.py`. No BLADE, solver,
  `graph_reward` formula, PPO, encoder, action-space, feature-width, detection-radius,
  hidden-placement, cardinality or executor-fuel-policy change.
  **Fix chain.** The FIRST candidate `1cf53fcee3ee05b3466c8391cbc6bb04420a0985` received
  REQUEST-FIXES on two measurement-honesty defects; the correction landed as a NEW CHILD
  COMMIT on the same branch and PR — never amend, rebase, force-push or history rewrite —
  and that fix commit touched FIVE of the seven files (`graph_fuel_damage.py`,
  `graph_tick_loop.py`, `graph_train.py` and the two test files). The two defects and
  their closure:
  (F1) the per-episode RTB output was derived from `GraphPlanExecutor.rtb_issued`, a
  LIFECYCLE LATCH that `_command_for_ego` also sets True for a DEAD ego — precisely
  because no command was emitted — so an ego that flew its plan into the ground counted
  as an RTB *and* a death. It is now taken from ACTUAL COMMAND HISTORY: `run_episode`
  hands each Phase-2 command list to `FuelDamageController.note_commands`, which latches
  only on a real `aircraft_return_to_base('<ego>')`.
  (F2) the preflight projection charges fuel for distance FLOWN while the engine burns
  `fuel_rate / 3600` every tick including route-less ones, so live fuel at the event is
  always below `projected_fuel_at_event` and the only guard was
  `fuel_before > post_damage_fuel`. The strict window is now RE-MEASURED from the live
  position through the same `measure_window` site and re-validated against live fuel
  BEFORE the mutation; a failure raises before anything is touched and is accounted as a
  `run`-stage failure, with planned and live bounds recorded under separate names.
  **Verified at the approved head:** full suite **192 passed, 4 skipped**,
  `tests/test_graph_fuel_damage.py` **35 passed**, `tests/test_graph_train.py`
  **73 passed**, import purity **12/12**, the `graph_trigger` selftest green, and
  `git diff --check` clean.
  **NO live BLADE/BONMIN probe, training run, rollout or scientific baseline was
  performed** — every test is solver-free and drives the pipeline through stubbed engine
  seams. Nothing in this lock is evidence about the cell's behaviour; §8 owns the gate.
- `24d1835` — **FINAL-CELL-VISUAL-ARTIFACTS: opt-in per-attempt inspection bundles —
  CLOSED / MERGED / REVIEWED.** Reviewed code SHA
  `24d1835f31d2e6aac04b418308a8753c392ac951`, integrated by merge commit
  `771f2107211fb3f984b64482b799613260e19aca` (PR #10); the merged tree is byte-identical
  to the approved one (`git diff --quiet 24d1835 771f210`). Grade A under `GPT_GITHUB`,
  implementation mode SURGICAL. The full technical contract is in §5 ("Visual artifacts —
  the opt-in inspection surface") and the routing in §6; this entry records the LOCK, not
  the mechanism.
  **Reviewed scope: EXACTLY TWO files** — `src/match_aou/rl/training/graph_train.py` and
  `tests/test_graph_train.py`. No generator, setup, tick-loop, fuel-damage, PPO, reward,
  solver, executor or vendored-BLADE file was touched, and no scenario semantics, seed
  formula, scenario-tag formula, scenario name, RNG draw, policy inference, PPO input,
  failure taxonomy or checkpoint/plot behaviour changed. `visual_artifacts` defaults to
  `False`, so a run that does not opt in is byte-unchanged.
  **Verified at the approved head:** `tests/test_graph_train.py` **89 passed** (73 → 89)
  under the base-env `pytest` AND all **89 passed** through the standalone `__main__`
  runner under `nlp_env`; `tests/test_graph_setup_seam.py` + `tests/test_graph_fuel_damage.py`
  **66 passed, 4 skipped**; import purity **12 passed**; full suite **208 passed,
  4 skipped** (192 → 208); `git diff --check` clean. Three mutation checks confirmed the
  load-bearing tests falsify (OFF path passing `recording_export_path=None`; the executed
  t=0 export moved after `build_fuel_damage_controller`; the `_VisualArtifactError`
  re-raise disabled) — each was caught and then reverted.
  **NEW FACT that shaped the implementation:** `tests/test_graph_fuel_damage.py` carries
  its OWN `_run_stub_training` with an independent `fake_run_one_episode` stub, so a new
  keyword passed UNCONDITIONALLY to `_run_one_episode` would break that unauthorized file.
  `_artifact_kwargs` therefore omits the keyword entirely on the OFF path — which is also
  the stronger invariance claim.
  **NO live BLADE/BONMIN probe, training run, rollout, artifact-generating smoke or
  scientific baseline was performed** — every test is solver-free and drives the pipeline
  through stubbed engine seams. Nothing in this lock is evidence about the cell's
  behaviour; §8 still owns the gate.

- `2a3f89c` — **Repository code hygiene: the retired minimal executor removed —
  CLOSED / MERGED / APPROVED.** Reviewed code SHA
  `2a3f89cf2d027581308493a98767ae658107d6d1`, integrated by
  `6e2757dd30100f429d492f4d23fd8b5f57cf4fac` (PR #11). Grade A under `GPT_GITHUB`,
  implementation mode SURGICAL — Grade A because it touched the locked
  `GraphPlanExecutor` and B2 route-prediction imports, even though the intended runtime
  behaviour was unchanged.
  `blade_executor_minimal.py` had stopped being an executor long ago and survived only
  because two live consumers imported its pure helper. **`nearest_neighbor_order` moved to
  `src/match_aou/utils/scheduling_utils.py`** — the environment-agnostic scheduling layer —
  with the body LINE-FOR-LINE unchanged, the only difference being the local type alias
  `Assignment` → that module's pre-existing, identical `Assignment3`. Both consumers
  (`GraphPlanExecutor._eligible`, `graph_hidden_placement.predict_route`) now import that
  ONE implementation, which IS the route-fidelity invariant. The retired file was deleted
  with **no shim and no re-export** (it remains on `flat-final`); stale prose was corrected
  in `blade_graph_executor.py`, `graph_hidden_placement.py` and `models/step.py` (the last
  still named the retired class as "the sole translation layer").
  Tests: `tests/test_executor_nn_ordering.py` → `tests/test_graph_executor_nn_ordering.py`,
  rewritten against current code (3 → 11 tests: the pure helper's order, exact tie-break,
  unlocated-last, no-anchor and end-location chaining, plus `GraphPlanExecutor` legacy-vs-NN
  ordering, live-position seeding, current-minimum-level gating and per-ego isolation).
  `tests/test_graph_hidden_placement.py` PO2 no longer uses the retired executor as its
  oracle: it consumes the CURRENT executor's own `_eligible` level by level from a live
  position, so the check stays independent of `predict_route` instead of becoming a
  tautological re-call of the shared helper.
  **Reviewed proof.** (i) HELPER SEMANTIC IDENTITY — empty normalized line diff against the
  base body. (ii) EXECUTOR / B2 FIDELITY — measured against the base SHA on both source
  trees: placement geometry byte-identical over 60 seeds × every `HiddenPlacement` field,
  the `geometric_fingerprint`, six predicted-route orderings and the post-placement RNG
  STREAM POSITION (so no draw was added, removed or reordered); executor eligibility
  byte-identical over 120 randomized worlds × both ordering modes × airborne / grounded /
  post-kill, plus emitted commands. Four mutation checks confirmed the load-bearing tests
  falsify — breaking `predict_route`'s chaining ALONE fails PO2, and so does breaking
  `_eligible`'s live-position seeding ALONE. (iii) NO DEPENDENCY LEAK — no Python or test
  import of `blade_executor_minimal` remains, `nearest_neighbor_order` has exactly one
  definition, and `graph_hidden_placement` purity IMPROVED (it no longer pulls in the
  `blade_utils` package at all).
  Verified: `tests/test_graph_executor_nn_ordering.py` **11 passed**,
  `tests/test_graph_hidden_placement.py` **18 passed**, import purity **12 passed**, full
  suite **216 passed, 4 skipped** (208 → 216), both `nlp_env` `__main__` runners green, and
  `git diff --check` clean. **NO training run, rollout, BONMIN solve, BLADE smoke or
  scientific probe was performed** — nothing here is evidence about the cell's behaviour;
  §8 still owns the gate.

- `52064c2` — **Repository / documentation hygiene — CLOSED / MERGED / APPROVED.**
  Approved candidate `52064c2d306df7c8447d159df20e6e189a59bf85`, integrated by
  `5f78904e3af1e2e47386c9b0e01ddbaa273724f5` (PR #12); the approved candidate tree was
  verified identical to the integration tree. Grade C under `GPT_GITHUB`, implementation
  mode SURGICAL. The FIRST candidate `6302847bdc8b5e40313763b4b167af85dd0a462e` received
  REQUEST-FIXES on two documentation-correctness findings — a backwards
  `Scenario.is_hostile` claim and stale volatile handoff state — and the correction landed
  as a NEW CHILD COMMIT on the same branch and PR, never amended, rebased or force-pushed.
  Scope: `README.md` replaced, `docs/BLADE_API_DOCUMENTATION.md` audited against the
  vendored fork, four obsolete scenario JSONs deleted, two dead utility symbols removed,
  the stale `requirements.txt` comment corrected, and `CLAUDE.md` / the handoff aligned —
  the §8 note below states the detail and is NOT repeated here. Verified: import purity
  **12 passed**, full suite **216 passed, 4 skipped** (unchanged from the base, since no
  runtime code changed), `git diff --check` clean. **NO scientific run, BONMIN solve,
  rollout, probe or artifact generation was performed.** After the merge the obsolete
  branches `task/repo-code-hygiene` and `task/repo-doc-hygiene` were deleted (safe
  deletion only, both already ancestors of `main`); `flat-final` and `pre-cleanup` were
  untouched, and every reviewed candidate tip remains reachable on GitHub through
  `refs/pull/<n>/head`.

- `61e539e` — **FINAL-CELL PROBE HARNESS: JSON presets, run layout and three semantic
  figures — CLOSED / MERGED / APPROVED.** Reviewed code SHA
  `61e539ed62fcf1e3fe25a83d213cae06f5afa98e`, integrated by merge commit
  `a5f389a2af328640e19db51d3277a33167c08f25` (PR #14); the merged tree is byte-identical
  to the approved one (`git diff` between the two reports zero changed files). Grade A
  under `GPT_GITHUB`, implementation mode SURGICAL — **the grade was corrected from B to A
  during review**, because `graph_train.py` is part of the §5 locked trainer contract; no
  implementation redo was required, since the strongest reasoning model, three proof
  obligations, the exact-SHA branch workflow and the broad test set were already in place.
  The full technical contract is in §5 ("Experiment harness") and the routing in §6; this
  entry records the LOCK, not the mechanism.
  **Reviewed scope: FOUR files** — `src/match_aou/rl/training/graph_train.py`,
  `tests/test_graph_train.py`, `README.md`, and the new
  `configs/graph_train/final_cell_probe.json`. No solver, BLADE, reward, fuel-damage,
  tick-loop, PPO, episode-setup, scenario-construction, seed-schedule, matched-pair
  evaluation or visual-artifact semantics were touched, and no evaluation RECORD FIELD was
  added, removed or redefined — the figures read fields that already existed.
  **Fix chain: TWO REQUEST-FIXES rounds, each landing as a NEW CHILD COMMIT on the same
  branch and PR** — never amended, rebased or force-pushed. Candidate
  `4238e0ee79faf3c1bde414fa041d410e44c07b38` → `de51883f20f28aadb4e6a9fa2a6f679a9eaded2f`
  → the approved `61e539e`. The five findings and their closure:
  (F1) `_explicit_cli_dests` read `argv=None` as an EMPTY command line while `argparse`
  reads it as `sys.argv[1:]`; since `main()` is normally called with no argument, every
  flag an operator really typed looked un-typed and a preset could silently override it.
  Both passes now consume one `_effective_argv` vector.
  (F2) `config_source` had two contradictory contracts (structured for CLI-only runs in
  code, `null` in the docs). Settled on ALWAYS-structured, one schema, one helper.
  (F3) the two held-out condition means are each over their own successful subset, so
  their gap is not a within-seed comparison; `measurement_health.png` gained PER-CONDITION
  completion counts and the performance panel's title and legends now say what each mean
  is over. Evaluation semantics and the matched-pair computation were NOT changed.
  (F4) the `config_source` fallback INFERRED `cli_defaults` from the absence of a path,
  which mislabelled every direct `train(cfg)` call — `_selftest` included — as
  CLI-resolved; `resolved_from` became a required argument and `direct_config` a third
  truthful kind.
  (F5) the preset's own prose promised a post-update round "after both updates", which the
  schedule cannot guarantee; it now says both SCHEDULED ITERATIONS and states that
  `updates_completed` may be 0, 1 or 2. PROSE ONLY — the schedule fields are byte-unchanged
  and test-pinned.
  **Verified at the approved head:** `tests/test_graph_train.py` **108 passed** (89 → 108),
  import purity **12 passed**, full suite **235 passed, 4 skipped** (216 → 235), all 108
  green through the standalone `__main__` runner under `nlp_env`, and `git diff --check`
  clean. Five mutation checks confirmed the load-bearing tests falsify (the `argv=None`
  reading; the unguarded `config_source`; the blanked per-condition series; the
  `cli_defaults` fallback; the restored "after both updates" prose) — each was caught and
  then reverted. Figures were rendered from SYNTHETIC records through the real `--plot`
  CLI and inspected.
  **NO BONMIN solve, BLADE episode, training run, rollout, selftest, probe or baseline was
  executed** — every test is solver-free and drives the pipeline through stubbed engine
  seams. **This lock certifies the HARNESS, not the cell**: no reward improvement and no
  fuel-damage behaviour has been measured on it. §8 still owns the gate.

- `d56fda6` — **DEFECT A: ego-global `SELF_PRESERVATION_ABORT` — CLOSED / MERGED /
  APPROVED.** Approved candidate SHA `d56fda636ab5ec1a5cce6076f07acac5556d10cb`,
  integrated by merge commit `f094e0b32e5e67b79757edbfe4e73c1fe01b0a87` (PR #17). The
  candidate was merged with a MERGE COMMIT and preserved as its SECOND PARENT; candidate
  and integration share the identical tree `70e5af2446f0a1b0674eb10819c9451753260560`, and
  the candidate→integration comparison contains ZERO changed files. Grade A under
  `GPT_GITHUB`, implementation mode SURGICAL. The technical contract is in §5 (Stage 4
  SELECTION, Stage 5 EFFECT); this entry records the LOCK, not the mechanism.
  **THE DEFECT.** The first executed bounded short probe (`training_output_20260815_173029`,
  from `238062d7d284334432d9c39d7543fb0bbf39ea7c`) showed `apply_meta_action` removing only
  the assignments whose `task_idx == node_v`, so SPA aborted ONE TASK rather than the ego's
  MISSION — playback showed a fuel-damaged KC-135 selecting SPA while its BLADE route
  continued and further assignments remained. The approved behaviour is an EGO-GLOBAL
  mission abort, reaching the ALREADY-EXISTING wake → resync → empty-plan → single-latched-RTB
  path. **The `k × 3` action head was NOT redesigned.**
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR.** The first candidate
  `c306455085de408c7bf383135c27e600ff3f1428` received REQUEST-FIXES for THREE
  documentation inaccuracies — a comment claiming the RTB is issued "on the next tick"
  (it is issued on the next `GraphPlanExecutor.next_actions` call, which is Phase 2 of the
  SAME tick), a stale `graph_fuel_damage` docstring still saying SPA would "drop the
  assignment", and a `MetaAction` docstring wrongly grouping `PLAN_COMPLIANCE` with
  `OPPORTUNISTIC_ENGAGEMENT` as acting on the selected node. The correction landed as a NEW
  CHILD COMMIT `d56fda6` — never amend, rebase, squash, force-push or history rewrite —
  and its non-docstring/non-comment token stream was verified identical to `c306455`.
  **CUMULATIVE SCOPE: EXACTLY FIVE FILES** — `src/match_aou/rl/action/graph_effect.py`
  (the sole runtime change), `src/match_aou/rl/action/graph_action.py` and
  `src/match_aou/rl/training/graph_fuel_damage.py` (both DOCUMENTATION-ONLY: their
  docstring text necessarily CHANGED, so their complete token streams are NOT identical —
  what was verified identical to the base is their NON-DOCSTRING / NON-COMMENT,
  runtime-relevant token stream), plus `tests/test_graph_fuel_damage.py` and
  `tests/test_graph_setup_seam.py`. No BLADE, executor, tick-loop, PPO, encoder, reward,
  solver, generator, scenario, seed-schedule, fuel-damage-mechanism, trainer, rollout,
  preset or artifact file was touched.
  **PROOF OBLIGATIONS.** PO1 — ego-global effect and private isolation: a multi-assignment
  actor across BOTH of its legal abort cells yields the identical empty slice, every peer
  slice is value-unchanged, the input `solution` and `tasks` are unmutated, out-of-range
  `node_v` still raises, and the real builder + real `build_action_mask` confirm the `k × 3`
  shape with abort legal on exactly the ego's own assigned nodes. PO2 — the REAL
  `graph_tick_loop._wake_decision` chain (real builder, mask, `sample_action`,
  `apply_meta_action` and `GraphPlanExecutor.resync`; only encoder/head stubbed to force a
  deterministic cell): before Phase 2 the actor's belief slice and executor plan are both
  empty while every peer is unchanged, then exactly ONE `aircraft_return_to_base`, no stale
  move/attack for that ego, and no second RTB toggle. PO3 — a solver-free REAL-BLADE tier:
  a real launched aircraft flying a real executor-issued mission route has `rtb` set, the
  stale waypoint removed and a route ending at its ACTUAL home base, with no second
  executor RTB; `Game.py` is byte-unchanged.
  **VERIFIED at the approved head:** full base suite **238 passed, 4 skipped** (235 → 238);
  focused base pytest (fuel damage, setup seam, action evaluate, import purity) **70 passed,
  4 skipped**; `tests/test_graph_fuel_damage.py` standalone `nlp_env` runner **37 passed**;
  `tests/test_graph_setup_seam.py` standalone `nlp_env` runner **20 passed, 0 skipped**,
  including the real-BLADE + BONMIN solver tier with no `CRASH`/`Traceback`; both
  action-layer selftests green under `nlp_env`; `git diff --check` clean. Falsifiability was
  demonstrated: with the old node-filtered body temporarily restored all four regressions
  fail and the `graph_effect` selftest fails at case (3); the mutation was reverted
  byte-identically and is not in the history.
  **NO scientific probe, training run, rollout or baseline was executed.** **This closes
  DEFECT A ONLY — as of THIS lock, Defects B and C both remained OPEN. Both have since
  been closed — B by `39a16f2` and C by `ea62e4e`, below; §8 owns the current state.**

- `39a16f2` — **DEFECT B: the attack-confirmation wait DERIVED from the salvo about to fly
  — CLOSED / APPROVED / MERGED.** Approved candidate SHA
  `39a16f2e5e1a3302d545c11b072e037e9702dffe`, integrated by merge commit
  `60a82d17398e9d14be1c2684cc72fafd020e0d9b` (PR #19). The candidate was merged with a
  MERGE COMMIT and preserved as its SECOND PARENT; candidate and integration share the
  IDENTICAL tree `ee86f0782ac50ee8bd0ee2fe634393a9cfc53a66` (verified locally), and the
  candidate→integration comparison contains ZERO changed files. Implementation fixed base
  `cefda78b18ea2daeda5014bab9a75a0945ef8e37`. Grade A under `GPT_GITHUB`, implementation
  mode SURGICAL. The technical contract is in §5 (Execution, Stage 1) and the routing in
  §6; this entry records the LOCK, not the mechanism.
  **THE DEFECT.** `GraphPlanExecutor` armed a FLAT `kill_confirm_ticks` for every salvo
  (default 60, and no caller passed it), so a slower auto-selected weapon could still be
  airborne when the wait expired — the executor then issued a redundant salvo that burned
  the ego's last weapons, measured in the first short probe's `post_update` damaged eval
  seed `1000003` (§8). The approved behaviour DERIVES the wait per salvo from the live
  auto-selected weapon and the current engagement distance, with the configured value kept
  as its FLOOR and FALLBACK. **The default was not merely raised, and frozen BLADE was not
  touched.**
  **CUMULATIVE SCOPE: EXACTLY TWO FILES** —
  `src/match_aou/utils/blade_utils/blade_graph_executor.py` and
  `tests/test_graph_setup_seam.py`. No vendored BLADE, solver, reward, PPO, encoder,
  action-space, tick-loop, trainer, rollout, fuel-damage, scenario, preset or artifact file
  was touched.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR.**
  (1) First candidate `45a0352312ae308df76a506a8e2e9907a9531a43` — the RUNTIME
  IMPLEMENTATION was ACCEPTED; GPT requested corrections because the transcribed
  `KILOMETERS_TO_NAUTICAL_MILES` was only compared against ANOTHER LITERAL rather than the
  engine's own constant, because the continuous-time bound was described inaccurately as
  exact engine ticks, and because the fallback prose contradicted the accepted
  negative-speed `abs` normalization.
  (2) The correction landed as the NEW CHILD COMMIT `39a16f2` — never amend, rebase,
  squash, force-push or history rewrite. Its RUNTIME-RELEVANT executor token stream is
  UNCHANGED from the first candidate; what it added is the real BLADE constant comparison,
  and what it corrected are the bound / timing / fallback claims.
  **PROOF OBLIGATIONS.** PO1 — the derivation really consults the LIVE BLADE selector
  `get_weapon_with_highest_engagement_range()`, the formula, the `max(…, bound + 1)` floor
  and every fallback branch behave as specified, and no peer aircraft, inventory, belief or
  assignment can move the acting ego's wait. PO2 — against the REAL engine: the
  redundant-salvo mechanism is exhibited by the flat-60 control arm and PREVENTED by the
  derived wait, the weapon reserve survives, and the confirmed-kill guard still advances the
  plan on the call the kill becomes visible (earlier than the bound), issuing RTB
  immediately. PO3 — the two-argument attack command, the per-`(ego_id, target_id)` cooldown
  identity, no-comms isolation and every frozen layer are preserved.
  **VERIFIED at the approved head:** full base suite **246 passed, 4 skipped**; focused
  suite **78 passed, 4 skipped**; standalone `tests/test_graph_setup_seam.py` under
  `nlp_env` **28 passed, 0 skipped** — the real-BLADE tier RAN, the engine-constant
  comparison RAN and the BONMIN tier RAN, with no `CRASH` and no `Traceback`; the executor
  selftest green; `git diff --check` clean.
  **NO scientific probe, training run, rollout or baseline was executed.** **This closes
  DEFECT B ONLY — as of THIS lock Defect C remained OPEN. It has since been closed by
  `ea62e4e` below; §8 owns the current state.**

- **DEFECT C: physical RTB completion — CLOSED / APPROVED / MERGED.** Approved candidate
  SHA `ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b`, integrated by merge commit
  `0de9f21eb9e8904f06f836f4ecd010bc46c788b6` (PR #21). The candidate was merged with a
  MERGE COMMIT and preserved as its SECOND PARENT (integration parents, in order:
  `6e97940733d2c7cf8c4ffc7033180c65f644ae17` then `ea62e4e…`); candidate and integration
  share the IDENTICAL tree `6d05cc5ea9af0f6bdcd4a2d6865767bcbe525ebe` (verified locally),
  and the candidate→integration comparison contains ZERO changed files. Implementation
  fixed base `6e97940733d2c7cf8c4ffc7033180c65f644ae17`. Grade A under `GPT_GITHUB`,
  implementation mode BUILD. The technical contract is in §4 (the pipeline's terminal
  loop) and §5 (Execution Stage 1, “COMPLETION IS PHYSICAL, NOT ISSUANCE”, and the
  tick-loop entry); the routing is in §6. This entry records the LOCK, not the mechanism.
  **THE DEFECT.** `GraphPlanExecutor.is_done()` read the `rtb_issued` lifecycle LATCH as
  RTB-resolved and `run_episode` stopped as soon as it became true, so an episode could
  end while the aircraft was still airborne — measured in the first short probe's
  `post_update` damaged eval seed `1000000`, which recorded `dead=0` and reward 0 for an
  ego that could not physically reach home. The approved behaviour separates “RTB command
  ISSUED” from “RTB physically RESOLVED” while PRESERVING the single-issue toggle guard.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend,
  rebase, squash, force-push or history rewrite. First candidate
  `5a0809df1a490df6ff266343788655d32fcefd81` (parent `6e97940…`) carried the runtime
  correction; the review correction landed as the NEW CHILD COMMIT `ea62e4e…` (parent
  `5a0809d…`), which names `is_done`'s two distinct sources explicitly and proves the
  burn-out branch directly.
  **CUMULATIVE SCOPE: EXACTLY SIX FILES** —
  `src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`, `tools/graph_executor_smoke.py`,
  `tests/test_graph_setup_seam.py` and `tests/test_graph_fuel_damage.py`. No vendored
  BLADE, solver, reward, PPO, encoder, action-space, trainer, rollout, fuel-damage
  mechanism, scenario, preset or artifact file was touched.
  **PROOF OBLIGATIONS.** PO1 — completion is decided from the LIVE post-step observation:
  issuance is not landing, a non-dead ego must be in an airbase inventory, an ego absent
  from both the air and every inventory is reconciled into `dead`, the reconciliation pass
  is total before any verdict, and the `add_return_to_base=False` contract is preserved.
  PO2 — against the REAL engine, the ride home reaches the terminal result: with fuel to
  spare the episode continues past the order and ends only on landing; with half the fuel
  the engine itself says the trip needs, the ego is removed mid-return, counted dead and
  charged by the unchanged reward path. PO3 — a returning ego is frozen out of Phase 1
  (no sensing, trigger, wake, inference, belief edit or transition; its belief is
  byte-frozen from the moment it commits) while peers continue normally, and no peer can
  decide whether the returning ego is home or lost.
  **ACCEPTED EVIDENCE.** Full base suite **257 passed, 4 skipped**; focused suite
  **89 passed, 4 skipped**; standalone `tests/test_graph_setup_seam.py` under `nlp_env`
  **35 passed, 0 skipped** — the real-BLADE tier RAN and the BONMIN tier RAN, with no
  `CRASH` and no `Traceback`; standalone `tests/test_graph_fuel_damage.py` **41 passed,
  0 failed**; the executor and tick-loop selftests green; `tools/graph_executor_smoke.py`
  reported **`SMOKE PASS`** with **3/3 egos physically returned to airbases**;
  `git diff --check` clean. The real-BLADE tier supplies BOTH lifecycle outcomes
  directly — a sufficient-fuel ego that lands, and an insufficient-fuel ego that dies —
  and the death is pinned by a DIRECT CAUSAL WITNESS on `Game.remove_aircraft`: exactly
  ONE recorded removal of that ego, at `current_fuel <= 0`, with no replacement airframe
  in any inventory (the landing branch would show positive fuel and an inventory entry, and
  a weapon kill bypasses `remove_aircraft` entirely, so it could leave no record at all).
  The returning-ego freeze and peer continuation are shown on the real `run_episode`.
  **NO scientific probe, training run, rollout or baseline was executed** — nothing in
  this lock is a corrected-cell measurement. **This closes DEFECT C ONLY**; with Defects
  A and B already closed, the three-defect CODE correction is complete, but that is an
  implementation fact, not a scientific result (§8 owns the gate).

- **ROSTER / WORLD-TRUTH INTEGRITY: executed-world inventory separated from oracle
  allocation — CLOSED / APPROVED / MERGED.** Approved candidate SHA
  `36365f210e8a659a641a7713f612c7e0ec1d4665` (`2026-08-17T14:01:10+03:00`), reviewed
  `APPROVE`, integrated by `f37ea1c8559405d5de24a9c2dd9e740227acaeeb`
  (`2026-08-17T15:48:30+03:00`, PR #24). **Candidate and integration share the IDENTICAL
  tree `f801538080f2ad282766d32346580189fa949f0c`, so the integrated tree is exactly the
  reviewed tree.** Grade A under `GPT_GITHUB`. The technical contract is in §5
  ("Roster / world-truth integrity" and the Stage-0 "WORLD INVENTORY IS NOT ORACLE
  ALLOCATION" block); the routing is in §6. This entry records the LOCK, not the mechanism.
  **THE DEFECT — a FOURTH, SEPARATE one, not a regression in Defects A, B or C.** The
  trainer answered "which targets does this episode contain?" from `ctx.beliefs` (known) and
  `ctx.oracle_tasks` (executed). Both are ALLOCATIONS: `solve_and_normalize` returns an
  allocated-only task list by contract, so every target the solver did not select was
  missing from them while still sitting in the world the executor flew through, sensed,
  attacked and confirmed. The roster therefore under-counted its own world and then FAILED
  the episode for the discrepancy it had itself introduced — **as an accounted `setup`
  failure**, which is why the long baseline above lost 143 of 800 training attempts to a
  measurement defect across 83 iterations while reporting itself healthy and reconciled.
  **THE APPROVED SEMANTICS.** `solve_and_normalize()` REMAINS allocated-only, and
  `belief_tasks` / `oracle_tasks` REMAIN allocations rather than world inventories —
  nothing about the oracle denominator changed. `known_target_ids` snapshots all raw
  known-world target ids before solver filtering; `executed_target_ids` snapshots all raw
  AUTHORITATIVE-world target ids before solver filtering; belief ids must agree across egos
  at t=0 and be a SUBSET of the known snapshot; hidden ids are executed MINUS known in
  executed-world order. The approved 3-known / 3-hidden / 6-total cell is checked
  (`_require_scheduled_cell`) BEFORE fuel-damage planning and before execution.
  Roster/world-integrity faults ABORT train and eval as
  INFRASTRUCTURE / DATA-INTEGRITY failures (`MeasurementIntegrityError`, with
  `EpisodeRosterError` as its subclass): they do NOT enter `EpisodeAttemptError`,
  `episode_failures.jsonl`, `skip_and_account_v1`, condition failure tallies, or any
  scientific denominator. After `run_episode`, playback synchronization
  (`_AttemptArtifacts.sync_recordings`) and confirmed-id validation happen BEFORE the
  reward; an `incomplete` manifest truthfully lists real playback that was already written;
  and a manifest cannot become `complete` when expected and observed world counts disagree.
  **Reward, PPO, oracle allocation, fuel damage, B2, seeds, schedules, the tick loop, the
  executor, the generator and vendored BLADE were UNCHANGED.**
  **REVIEWED SCOPE: FIVE files** — `src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_setup_seam.py`,
  `tests/test_graph_train.py`, `tests/test_graph_fuel_damage.py`.
  **ACCEPTED IMPLEMENTATION EVIDENCE:** focused base-environment suite **207 passed, 4
  skipped**; full suite **272 passed, 4 skipped**; standalone `tests/test_graph_train.py`
  under `nlp_env` **119 passed**; standalone `tests/test_graph_fuel_damage.py` **41
  passed**; standalone `tests/test_graph_setup_seam.py` **39 passed, 0 skipped**, including
  the real-BLADE and BONMIN tiers; `git diff --check` clean.
  **NO training run, probe, rollout, seed sweep or baseline rerun occurred during the
  correction.** **CONSEQUENCE FOR THE TWO AFFECTED MEASUREMENTS:** the long baseline above
  is `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED`, and the corrected-cell short probe's
  `VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS` verdict is SUPERSEDED by
  `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE SCIENTIFIC
  DENOMINATOR`. This lock certifies the CODE correction; it is **not** a measurement of the
  cell, and no result may be pre-claimed for the rerun §8 then authorized. **That rerun has
  since been EXECUTED, independently reviewed and APPROVED — see the VALID long-baseline
  entry below, which is the authoritative measurement of this cell.**

- `eecc9b5` — **FD-VARIABLE-SEVERITY-v1: the mild/severe fuel-damage research factor with
  matched clean/mild/severe evaluation — CLOSED / APPROVED / MERGED.** Approved candidate
  SHA `eecc9b5d91bce4a98a070a29307cc12af0d4c4a3`, integrated by merge commit
  `177e969446ef6c01c729484f2ea9969c94a27330` (`2026-08-20 12:15:28 Asia/Jerusalem`,
  PR #27). The candidate was merged with a MERGE COMMIT and preserved as its SECOND
  PARENT (ordered parents: `4f0068847b017795717c5f0e331f647bcfc30547`, then
  `eecc9b5…`); candidate and integration share the IDENTICAL tree
  `37ebd8c56266fdd862cc7244c5f22a6ac95e438c` (verified locally), and the
  candidate→integration comparison contains ZERO changed files. Grade A under
  `GPT_GITHUB`. The technical contract is in §5 (the FD-VARIABLE-SEVERITY-v1 mechanism
  block, its measurement-surface block, and the scheduled-vs-executed cell block) and the
  routing in §6. This entry records the LOCK, not the mechanism.
  **THE RESEARCH PROBLEM.** Under the merged FD-BASELINE-v1 design EVERY damaged episode
  is structurally SEVERE, so "damaged" and "continuing is infeasible" are the SAME fact
  and a trained actor can learn the shortcut `fuel damage ⇒ abort` without ever reading
  its own fuel gauge. The approved extension splits the damaged half into a MILD band
  where continuing remains genuinely feasible and a SEVERE band where it does not, so the
  response has to be read off the ego's own live fuel. **The LEGACY modes are UNCHANGED —
  same seeds, same conditions, same selected egos, same planned-midpoint target — because
  an approved measurement exists on them** (`737b4bf`, the entry above), and a factor that
  moved them would invalidate that baseline instead of extending it.
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. First candidate
  `73752d872a8cd17f703790ef41bee46a734170bb` (parent `4f00688…`) received REQUEST-FIXES on
  ONE measurement-integrity defect; the correction landed as the NEW CHILD COMMIT
  `eecc9b5…` (parent `73752d8…`), touching TWO files
  (`src/match_aou/rl/training/graph_train.py`, `tests/test_graph_fuel_damage.py`).
  **THE FINDING AND ITS CLOSURE.** `_ConditionTally.success` checked only that the EXECUTED
  cell was a LEGAL cell of the run. Under the new design that membership test ACCEPTS a
  scheduled `mild` that executed as `severe` — booking the ATTEMPT in one cell's
  denominator and the REWARD in another, corrupting BOTH at once, and letting a triad's
  within-seed delta be taken between two members the schedule never paired. The approved
  fix makes `expected_cell` a REQUIRED keyword and requires scheduled == executed
  EQUALITY before ANY accounting, with both production call sites (training and
  evaluation) passing their scheduled cell and the guard running FIRST — so a mismatched
  episode reaches neither the tally, nor a matched-group member reward or delta, nor
  `episode_outcomes.jsonl`, nor the PPO buffer. A mismatch is a
  `MeasurementIntegrityError` INFRASTRUCTURE abort, never an accounted scientific episode
  failure. **It has NOT been observed in the real simulator**: the regression test INJECTS
  the divergence through a stub, because normal production does not currently generate it.
  **CUMULATIVE REVIEWED SCOPE: EXACTLY FIVE FILES** —
  `src/match_aou/rl/training/graph_fuel_damage.py`,
  `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_rollout.py`, `tests/test_graph_fuel_damage.py` and
  `tests/test_graph_train.py`. No vendored BLADE, solver, `graph_reward`, PPO, encoder,
  action-space, tick-loop, executor, episode-setup, hidden-placement, generator, scenario,
  preset, config or README file was touched. Target destruction remains DETERMINISTIC at
  `probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate
  future Grade-A research task** (§8).
  **ACCEPTED EVIDENCE at the approved head:** `tests/test_graph_fuel_damage.py`
  **60 passed**; `tests/test_graph_train.py` **119 passed**; both standalone `nlp_env`
  `__main__` runners **60 passed / 119 passed**; full suite **291 passed, 4 skipped**;
  `git diff --check` clean. `graph_train --selftest` — TEST 1 passed, TEST 2 passed, and
  **TEST 3 failed IDENTICALLY TO THE BASE** on the already-known B2 seed-2
  exact-cardinality case (§8): a pre-existing expected outcome of the current contract,
  **not a PR #27 regression**.
  **NO scientific baseline, long training run, probe, rollout or artifact-generating smoke
  was executed for PR #27** — that remains true of THIS lock, which certifies CODE only.
  Nothing in this entry is a measurement of the variable-severity cell. **The measurement
  was taken separately and afterwards**, at measured code SHA `bf1e045f`, and it is
  recorded in the next §7 entry; §8 owns the phase state.
- `a6f3aa9` — **PHASE-B CTDE: the centralized critic during TRAINING only, with
  `actor_only` preserved as the default — CLOSED / APPROVED / MERGED.** Approved candidate
  SHA `a6f3aa9d62931994f416b2241fec4cfac3b018ec` (`2026-08-22 21:01:46 Asia/Jerusalem`),
  integrated by merge commit `8390d85c2072e9cbe984ce5f2731cef3a9b14985` (PR #30). The
  candidate was merged with a normal MERGE COMMIT and preserved as its SECOND PARENT
  (ordered parents: `d437084c5fb1a22c21596a48c58e03f7e15a0115`, then `a6f3aa9…`), and the
  integration tree is `9686c107b8864f00a7d4403d70faf42ab561d2fb`. **Grade A under
  `GPT_GITHUB`, implementation mode BUILD** — it created a new layer and a new module, and
  the SURGICAL mode belongs to the SEPARATE documentation-lock task that recorded it, never
  to the implementation itself. The technical contract is in §5 ("PHASE-B CTDE — the
  TRAINING-ONLY centralized critic"), the pipeline placement in §4 and the routing in §6.
  This entry records the LOCK, not the mechanism.
  **THE TWO IMMUTABLE REFERENCES, AND THEY ARE DISTINCT.**
  `pre-ctde-actor-only = d437084c5fb1a22c21596a48c58e03f7e15a0115` (tree
  `d7cc2dcb1b161180e272afc9600175f022c5b5d0`) is the NEW immutable reference preserving the
  IMMEDIATE PRE-CTDE actor-only state — it is the integration's FIRST parent, so "the
  actor-only state CTDE was merged onto" is a git fact rather than a claim. Preserving it
  was the CTDE integration gate's remaining prerequisite (§8), and it must not move.
  `phase-a-baseline = 4f0068847b017795717c5f0e331f647bcfc30547` is the SEPARATE, ORIGINAL
  Phase-A reference, is NOT repurposed as the pre-CTDE reference, and likewise must not
  move. Neither is the FD-VARIABLE-SEVERITY-v1 measured code SHA
  `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, which is a durable MEASUREMENT identity and
  never a code reference (its `APPROVE — VALID MEASUREMENT` record, with its NEGATIVE
  primary finding, is above and is UNCHANGED by this lock).
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. Initial reviewed candidate
  `d70d07f829a44e6f19100c338d4dde89f4f47bf6` (`2026-08-22 20:02:21 Asia/Jerusalem`) carried
  the implementation; the review correction landed as the NEW CHILD COMMIT `a6f3aa9…`,
  which is the APPROVED head. The three findings and their closure:
  (F1) `training_mode='ctde'` accepted `value_coeff == 0`. Such a run would build central
  observations and take its advantages from a critic it never trains, leaving the baseline
  a frozen random function — neither the `actor_only` reference algorithm nor the approved
  CTDE one, and recorded and read as CTDE either way. `TrainConfig.validate` now REQUIRES
  `> 0`, refused before any compute; the default is unchanged at `0.5`, `actor_only`
  validation is untouched, and `value_coeff` is still NOT a mode selector.
  (F2) CTDE training records did not persist the critic's diagnostics. `value_loss`,
  `value_mean`, `value_target_mean` and `critic_grad_norm` are now copied straight out of
  the dict `CTDEUpdater.update` already returned — never recomputed — and added ONLY when
  `ctde_enabled`, so an `actor_only` record is byte-unchanged with those keys ABSENT rather
  than null. No actor-side metric changed meaning.
  (F3) `ValueHead`'s docstring claimed a small init giving an initial value function "~0
  everywhere" while the code passes `std=1.0`. **PROSE ONLY — THE INITIALIZATION IS
  UNCHANGED**: `graph_ppo.py`'s runtime token stream (comments and docstrings stripped) is
  IDENTICAL to the initial candidate's.
  **REVIEWED SCOPE: EXACTLY SIX FILES**, verified as the complete
  `d437084…...8390d85…` comparison —
  `src/match_aou/rl/observation/central_graph_builder.py` (new),
  `src/match_aou/rl/training/graph_ppo.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_ctde.py` (new) and
  `tests/test_graph_ppo.py`. **NO documentation file was part of the code integration** —
  that is what this documentation task closes. No vendored BLADE, solver, `graph_reward`,
  `graph_fuel_damage`, encoder, action-space, episode-setup, hidden-placement, generator,
  scenario, config or preset file was touched, and the Phase-A cell, the seed schedules,
  the evaluation design, the failure taxonomy and every preserved run artifact are
  unchanged.
  **PROOF SURFACE** (`tests/test_graph_ctde.py`, and the import/dependency contract in
  `tests/test_graph_ppo.py`): actor-only preservation under a POISON that raises at every
  central-CTDE construction site, with a CONTROL proving the poison fires under `ctde`;
  disjoint actor/critic parameter sets; each backward leaving the other side's gradients
  `None`; the actor advantage a detached scalar; privileged central features unable to move
  an actor logit; a central state rejected as an actor observation; the `NO_EGO_INDEX`
  role symmetry; evaluation constructing neither critic nor recorder; a CTDE-trained actor
  running with the critic absent; hand-computed GAE, the zero terminal next value, the
  per-episode boundary and value targets fixed across epochs; `baseline` proven to be the
  mean episode REWARD and not the critic value; zero-wake handling; loud failure on
  misaligned central samples; variable graph sizes finite without padding; the exactly-five
  actor-only checkpoint keys beside the CTDE payload; and the persisted critic diagnostics
  with their absence on the `actor_only` path.
  **CC-REPORTED ENGINEERING EVIDENCE — IMPLEMENTATION VALIDATION, NOT SCIENTIFIC
  EVIDENCE. It has TWO parts, and they are labelled separately because they are different
  kinds of evidence.**
  (i) **TESTS — solver-free, stubbed engine seams.** At the approved head: full solver-free
  suite **334 passed, 4 skipped**; `tests/test_graph_ctde.py` **43 passed**;
  `tests/test_graph_train.py` **119 passed**; `tests/test_graph_ppo.py` **18 passed**; the
  standalone `nlp_env` CTDE `__main__` runner **43 passed**; `git diff --check` clean. Four
  mutation checks confirmed the fix-commit regressions falsify (the permissive
  `value_coeff` bound, dropped persistence, recomputed-instead-of-copied values, and CTDE
  keys leaking onto the `actor_only` path), each reverted.
  (ii) **BOUNDED ENGINEERING SMOKES — REAL BLADE AND REAL BONMIN, and they DID happen.**
  During the BUILD candidate's validation, **TWO bounded smokes under `nlp_env` ran BOTH
  training modes end-to-end against the real engine and the real solver**: 2/2 episodes,
  one PPO update, `accounting_reconciled = true`, no `CRASH` and no `Traceback`, writing
  only to the scratchpad and never into the repository. They are ENGINEERING evidence that
  the wiring executes, and they are what surfaced the `baseline`-vs-critic-value defect the
  contract now pins (§5). **Their rewards and episode outcomes are NOT scientific evidence
  and must never be promoted into any**, and the later append-only review-fix validation
  needed no new run of them.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #30** — no baseline, no
  probe, no scientific rollout, and above all **no actor-only vs CTDE comparison. NO CTDE
  benefit is established or may be pre-claimed.** Two bounded engineering smokes are not a
  measurement: they have no scientific contract, no seed schedule, no held-out band and no
  denominator. This lock certifies the IMPLEMENTATION; §8 owns the gate and the next
  scientific task.

- `5b55ca3` — **GENERALIZED-V1 TASK 1: generalized construction cardinality, deterministic
  bounded B2 backoff, and truthful requested-vs-realized accounting — CLOSED / APPROVED /
  MERGED.** Approved candidate SHA `5b55ca348309b4241d2087c2f60327bc842ea6fa`, integrated by
  merge commit `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9` (PR #35), from fixed base
  `7b86098a7573be15b0d8bfcf959b1d1f63288ffc`. Grade A under `GPT_GITHUB`, implementation
  mode BUILD. The technical contract is in §5 ("GENERALIZED-V1 HIDDEN CARDINALITY") and the
  routing in §6; this entry records the LOCK, not the mechanism.
  **WHAT IT IMPLEMENTS, AND WHAT IT DELIBERATELY DOES NOT.** It implements the generalized
  cardinality and B2 backoff direction ONLY (handoff §3l.1 / §3l.2). **NOT here:** FD
  eligibility-by-construction, persistent post-FD state or repeated wakes, the
  event-conditioned continuation reference, any reward change, the generalized training
  sampler, evaluation manifests, `p(destroy) < 1`, or multiple hidden targets on one route.
  `graph_train.py` was NOT touched.
  **REVIEWED SCOPE: EXACTLY FOUR FILES** —
  `src/match_aou/rl/training/graph_hidden_placement.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`,
  `tests/test_graph_hidden_placement.py` and `tests/test_graph_setup_seam.py`. No vendored
  BLADE, solver, reward, PPO, encoder, action-space, tick-loop, executor, fuel-damage,
  trainer, rollout, generator, config, preset or README file was touched, and no scenario
  semantics, seed formula or evaluation record field changed. `exact_v1` is the DEFAULT, so
  a caller that does not opt in is unchanged.
  **PROOF OBLIGATIONS.** PO1 — HISTORICAL PATH PRESERVATION: the exact path's geometry,
  chosen legs, ego order, sampled fractions, sampled offsets AND the episode rng's post-call
  STREAM POSITION are pinned to values captured from the pre-generalized implementation at
  the base commit over four seeds, so one added, missing or reordered draw fails; its loud
  no-route and unusable-leg refusals are pinned separately; `n_hidden=0` stays legal; the
  exact construction path carries no audit and grows no `split_meta` key. PO2 —
  DETERMINISTIC BOUNDED BACKOFF: ordinal-driven and not id text (two rosters with opposite
  lexical orders and the same ordinal→route mapping give identical results); the same seed
  reproduces order, selection, audit and fingerprint; a candidate's failure cannot shift a
  later candidate's geometry; the episode rng's end position depends only on the candidate
  count; short realization is reported truthfully with named reasons; zero realization is
  refused; no route is used twice over a 40-seed sweep; solver-omitted egos are still
  candidates and are recorded `no_route`. PO3 — SETUP / WORLD-TRUTH ACCOUNTING, under real
  bonmin at `(A=3, H=3)` and `(A=4, H=2)`: at reference seed 2 — the case §8 documents,
  where the static solve leaves one ego routeless — the EXACT path refuses the world while
  the BOUNDED path accepts it at `H_realized = 2/3` and names the routeless candidate; the
  patch adds exactly `H_realized` targets; env-2 stays authoritative; the oracle allocation
  is a subset of the world and never its inventory; reproducibility is by geometry while
  every hidden uuid differs.
  **CC-REPORTED ENGINEERING EVIDENCE at the approved head** — base-env full suite
  **359 passed, 6 skipped** (334 → 359; the two new solver-tier tests account for the added
  skips); `nlp_env` `tests/test_graph_hidden_placement.py` **33 passed** (18 → 33); `nlp_env`
  `tests/test_graph_setup_seam.py` **51 passed, 0 skipped** (39 → 51) — the real-BLADE tier
  RAN and the BONMIN tier RAN, with no `CRASH` and no `Traceback`; `git diff --check` clean.
  **Eight mutation checks** confirmed the load-bearing tests falsify (id-text ordering;
  substreams removed; one extra exact-path draw; accepting zero realized; silently rewriting
  the request down; judging the cell after the solve; trusting the audit instead of verifying
  it; silently ignoring the policy on the legacy path), each reverted.
  **NO training run, rollout, probe, scientific smoke or measurement of any kind was
  executed.** This lock certifies CODE only; §8 owns the phase state.

- `185d39f` — **GENERALIZED-V1 TASK 2: certified FD eligibility + post-FD
  completion-boundary adaptation — CLOSED / APPROVED / MERGED.** FINAL approved candidate
  SHA `185d39f00335a0bb5e9130cc773da94c914f17f5`, integrated by merge commit
  `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b` (PR #36), from fixed base
  `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9`. Grade A under `GPT_GITHUB`, implementation
  mode BUILD. The technical contract is in §5 ("GENERALIZED-V1 CERTIFIED FD ELIGIBILITY +
  POST-FD COMPLETION-BOUNDARY ADAPTATION", plus the Stage-1 one-confirmation-site block and
  the Stage-2 trigger entry) and the routing in §6; this entry records the LOCK, not the
  mechanism.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. Initial reviewed candidate
  `2f9231d989acf30561ecf10e74cf0c5491771836` received **REQUEST FIXES**; the correction
  landed as the NEW CHILD COMMIT `185d39f…`, which is the APPROVED head. **THE FINDING AND
  ITS CLOSURE:** the certified promise was guarded only from the INSIDE — `maybe_apply`
  aborted when the run REACHED the event and the live state contradicted the certificate,
  but a certified DAMAGED episode that simply ENDED without the event ever firing was
  returned as an ordinary successful damaged episode. That would admit a world whose
  certificate did not materialize into a scientific population. The fix adds
  `FuelDamageController.require_certified_event_realized`, called ONCE at the single
  `graph_tick_loop.run_episode` episode-exit seam and BEFORE the recording export, raising
  `FuelDamageIntegrityError` in exactly one of its four cells; a certified CLEAN episode
  legitimately finishes with `fired == False`, and the LEGACY policy returns ALWAYS —
  because an approved measurement (§7, the Phase-A rerun's seed 424) contains exactly such a
  non-firing damaged episode, and adding a terminal requirement to the legacy path would
  change the behaviour that measurement was taken on rather than extend it. The fix commit
  touched FOUR files (`graph_fuel_damage.py`, `graph_tick_loop.py`, and the two test files).
  **CUMULATIVE REVIEWED SCOPE: EXACTLY NINE FILES** —
  `src/match_aou/rl/training/graph_fuel_damage.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`, `src/match_aou/rl/action/graph_trigger.py`,
  `src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_fuel_damage.py`,
  `tests/test_graph_executor_nn_ordering.py`, `tests/test_graph_ctde.py` and
  `tests/test_graph_train.py`. No vendored BLADE, solver, `graph_reward`, PPO, encoder,
  action-space, episode-setup, hidden-placement, generator, rollout, scenario, config,
  preset or README file was touched. `graph_train`'s change is the exception ROUTING and
  nothing else. **Implements handoff §3l.3 / §3l.4 ONLY:** no continuation reference, no
  `U_prefix`, no reward change, no generalized training sampler, no evaluation manifest, no
  new metric or plot, no `p(destroy) < 1`, no new `MetaAction`, no trim-tail action.
  **PROOF SURFACE.** Both policy seams default to the merged behaviour and that default is
  test-pinned through `TrainConfig.fuel_damage_parameters()` and
  `RolloutConfig.fuel_damage_parameters()`; the three RNG domains are proven independent;
  candidate ordering is proven ordinal-driven and not id-text-driven; the certified walk is
  proven to run for CLEAN as well and to certify the same ego for all three matched members;
  the tick-aware prediction is checked against the frozen engine's own arithmetic; the
  executor's extracted `_reconcile_confirmed` is proven to reproduce the historical
  confirm-and-advance behaviour, to stay proximity-gated and peer-free, to ignore a dead or
  grounded ego, to still block on an unexecutable head, and to leave the tick's emitted
  command BYTE-IDENTICAL; and `tests/test_graph_ctde.py` proves a boundary capture follows
  the local reconciliation while central samples stay exactly 1:1 with actor transitions,
  with a control showing the DEFAULT wake policy produces no boundary sample at all.
  **CC-REPORTED ENGINEERING EVIDENCE, IN TWO LABELLED PARTS.**
  (i) **TESTS — solver-free where stated, plus the real-engine tiers.** At the approved head:
  base-env full suite **396 passed, 6 skipped** (365 → 402 collected);
  `tests/test_graph_fuel_damage.py` **85**, `tests/test_graph_train.py` **119**,
  `tests/test_graph_ctde.py` **45**, `tests/test_graph_executor_nn_ordering.py` **18**,
  import purity **12/12**; standalone `nlp_env` runners 85 / 119 / 45 / 18, plus
  `tests/test_graph_setup_seam.py` **51 passed, 0 skipped** with the real-BLADE and BONMIN
  tiers running and no `CRASH` or `Traceback`; the `graph_trigger` and
  `blade_graph_executor` selftests green; `git diff --check` clean.
  (ii) **ONE BOUNDED ENGINEERING SMOKE — real BLADE and real BONMIN under `nlp_env`.** A
  generalized `bounded_backoff_v1` world at `A=3 / K=3 / H_requested=3 → H_realized=3`,
  seed 0: clean, mild and severe all certified ordinal `1` — the same ego and a
  byte-identical certificate, event tick 137 — and live the event fired at tick **137**
  (certified 137 ± 1) with `|Δfuel_before| = 0.000000`, MILD continuation margin
  **+2695.77** and SEVERE **−939.60**, with no `FuelDamageIntegrityError`. It wrote only to
  a scratchpad and never into the repository. **IT IS IMPLEMENTATION VALIDATION, NOT A
  MEASUREMENT** — no policy was trained, no research directory was created, and no number in
  it is a scientific result; its rewards and outcomes must never be promoted into one.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #36** — no baseline, no probe,
  no scientific rollout, and **no generalized measurement existed at this lock, and no
  generalized result may be pre-claimed** (§8 owns the live run state). A
  bounded smoke is not a measurement: it has no scientific contract, no seed schedule, no
  held-out band and no denominator. This lock certifies the IMPLEMENTATION; §8 owns the
  phase state and the next task.

- `24a8b1e` — **GENERALIZED-V1 TASK 3: event-conditioned MATCH-AOU continuation reference +
  reward checkpoint — CLOSED / APPROVED / MERGED.** Reviewed candidate SHA
  `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integrated by merge commit
  `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25` (`2026-08-25 22:08:34 Asia/Jerusalem`, PR #38),
  from fixed base `ca2fe346b5fb5d499b6a59b3da17c74b2a8bae8e`. The candidate was merged with a
  normal MERGE COMMIT and preserved as its SECOND PARENT (ordered parents: `ca2fe346…`, then
  `24a8b1ee…`); candidate and integration share the IDENTICAL tree
  `187aed9105eca5db799f4508374dc86811001b9d` (verified locally), and the
  candidate→integration comparison contains ZERO changed files — so the integrated tree is
  exactly the reviewed tree. Grade A under `GPT_GITHUB`, implementation mode BUILD, verdict
  **APPROVE**. The technical contract is in §5 ("GENERALIZED-V1 EVENT-CONDITIONED MATCH-AOU
  CONTINUATION REFERENCE + REWARD CHECKPOINT", plus the Stage-0 and Stage-7 cross-references),
  the pipeline placement in §4 and the routing in §6; this entry records the LOCK, not the
  mechanism.
  **WHAT IT IMPLEMENTS, AND WHAT IT DELIBERATELY DOES NOT.** It implements handoff §3l.5
  ONLY — one OPT-IN reward-reference policy beside the historical one, which stays the
  DEFAULT and is untouched. **NOT here:** the generalized training sampler, the frozen
  stratified evaluation manifest, any run-level persistence or aggregate metric for
  `EpisodeReference`, any plot, `p(destroy) < 1`, any new `MetaAction`, and any change to
  terminal credit placement, PPO, GAE, the encoder, the action space, the trigger layer, the
  fuel-damage mechanism, the seed schedules, the frozen solver or vendored BLADE.
  **REVIEWED SCOPE: EXACTLY FIVE FILES** —
  `src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_reward.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `tests/test_graph_reference_continuation.py` (new) and `tests/test_graph_setup_seam.py`.
  **`graph_train.py` and `graph_rollout.py` were NOT touched**, and no config, preset,
  benchmark, README or documentation file was part of the code integration — that is what
  this documentation task closes. `static_t0_v1` is the DEFAULT, so a caller that does not
  opt in is unchanged.
  **THE ONE APPROVED IMPLEMENTATION DEVIATION, RECORDED RATHER THAN HIDDEN:** the reference
  kind `damaged_event_unrealized_t0`. The GPT review examined it and APPROVED it as a
  COMPATIBILITY RESOLUTION — it exists solely to preserve the already-locked Task-2 LEGACY
  contract, under which a scheduled damaged episode may legitimately finish without the FD
  event firing (§7: the Phase-A rerun's seed 424). **It is NOT the intended GENERALIZED
  damaged semantics and must not be generalized into them**: under
  `certified_both_severities_v1` it is unreachable, because the terminal
  `require_certified_event_realized` rejects certified + damaged + not-fired first (§5).
  **CC-REPORTED ENGINEERING EVIDENCE, IN TWO LABELLED PARTS — IMPLEMENTATION VALIDATION, NOT
  SCIENTIFIC MEASUREMENT.**
  (i) **TESTS.** Base-env full suite **430 passed, 6 skipped** (403 + 27 new, no existing
  count moved); the new solver-free and BLADE-free `tests/test_graph_reference_continuation.py`
  **27 tests** over PO1/PO2/PO3, green under base `pytest` AND the standalone `nlp_env`
  runner; standalone `nlp_env` runners `test_graph_fuel_damage.py` **91**,
  `test_graph_ctde.py` **45**, `test_graph_train.py` **123**, `test_graph_setup_seam.py`
  **51 passed, 0 skipped** (real BLADE + BONMIN tiers); the `graph_reward` and
  `graph_episode_setup` selftests green; **10 mutation checks** confirmed the load-bearing
  tests falsify, each reverted.
  (ii) **ONE BOUNDED ENGINEERING SMOKE — real BLADE and real BONMIN.** Both opt-in conditions
  plus a historical control, 2/2 solves each, both terminations `optimal`, with the damaged
  solve taken at the event tick over the live post-mutation fuel and position. **IT IS
  IMPLEMENTATION VALIDATION, NOT A MEASUREMENT** — no policy was trained, no research
  directory was created, and no number in it is a scientific result; its outcomes must never
  be promoted into one.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #38** — no baseline, no probe,
  no scientific rollout, and **no generalized measurement existed at this lock, and no
  generalized result may be pre-claimed** (§8 owns the live run state). A
  bounded smoke is not a measurement: it has no scientific contract, no seed schedule, no
  held-out band and no denominator. This lock certifies the IMPLEMENTATION; §8 owns the phase
  state and the next task.

- `db79013` — **GENERALIZED-V1 TASK 4: the episode-design selector, the generalized training
  cardinality sampler, the frozen stratified benchmark manifest and run-level persistence —
  CLOSED / APPROVED / MERGED.** FINAL approved candidate SHA
  `db79013897a6e5669f50d53b6e30229b16aea28d` (committed `2026-08-26 11:03:42 +0300`),
  integrated by merge commit `b4daa8c1a8c870061b26cceb01d4ed34169594e7` (`2026-08-26
  11:30:07 +0300`, PR #40), from fixed base `f4e8d3b8ddc61525fe0cde6b61ca4d611ebd2eed`. The
  candidate was merged with a normal MERGE COMMIT and preserved as its SECOND PARENT (ordered
  parents: `f4e8d3b8…`, then `db790138…`); candidate and integration share the IDENTICAL tree
  `f7cfd5cb2a551bddd5bfecf78fdcc83e2dcedef7` (verified locally), and the
  candidate→integration comparison contains ZERO changed files — so the integrated tree is
  exactly the reviewed tree. Grade A under `GPT_GITHUB`, implementation mode BUILD, verdict
  **APPROVE**. The technical contract is in §5 ("GENERALIZED-V1 EPISODE-DESIGN SELECTOR,
  TRAINING CARDINALITY SAMPLER, FROZEN STRATIFIED BENCHMARK MANIFEST AND RUN-LEVEL
  PERSISTENCE", plus the corrected `ReferenceIntegrityError` routing block and the three
  harness-exposure corrections in the Task-1/2/3 blocks), the selector's pipeline placement is
  in §4, and the routing is in §6. This entry records the LOCK, not the mechanism.
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. The original reviewed candidate
  `eef1795f6bb3f0cbc4c163ba489cf5e790df4c41` (`2026-08-26 10:16:25 +0300`) carried the
  implementation; the review correction landed as the NEW CHILD COMMIT `db790138…`, which is
  the APPROVED head and touched FOUR files
  (`src/match_aou/rl/training/graph_generalized.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_generalized.py`,
  `tests/test_graph_train.py`) across three named corrections: **manifest integrity** (the
  loader now authenticates the EXACT STORED payload *and* independently requires it to equal
  the canonical payload, so a self-consistently rehashed noncanonical forgery is refused as
  well as a tampered one), **real held-outness** (`manifest_seed_overlap` /
  `_require_benchmark_seeds_held_out` check the manifest's ACTUAL seeds against the training
  band at LOAD time — before the run directory or any scientific compute — instead of leaning
  on a configured band the run never executes, which could have falsely rejected a held-out
  manifest or falsely validated one containing a training seed), and **honest construction
  provenance** (a `generalized_v1` `run_config.json` no longer writes the unused configured
  3/3/3 counts in the historical shape, which would have read as "this run executed 3/3/3";
  the cell is declared DYNAMIC, the configured numbers survive only under an explicitly
  labelled `unused_fixed_cell_config` marked `executed: false`, and realized per-episode
  counts are pointed at `episode_outcomes.jsonl`).
  **CUMULATIVE REVIEWED SCOPE: EXACTLY EIGHT FILES**, verified as the complete
  `f4e8d3b8…...b4daa8c1…` comparison — `src/match_aou/rl/training/graph_generalized.py`
  (NEW, 1386 lines), `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_rollout.py`, `src/match_aou/rl/training/graph_reward.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`, `tests/test_graph_generalized.py`
  (NEW), `tests/test_graph_train.py` and `tests/test_graph_reference_continuation.py`. **NO
  documentation file was part of the code integration** — that is what this documentation task
  closes — and **no config, preset or benchmark manifest was added or changed**
  (`configs/graph_train/final_cell_probe.json` remains the ONLY repository preset, and it is
  untouched and still `fixed_cell_v1`). No vendored BLADE, solver, PPO, encoder,
  action-space, tick-loop, executor, fuel-damage-mechanism, hidden-placement or generator file
  was touched. `graph_reward`'s change is the reason-carrying `ReferenceIntegrityError` and
  its `reference_fault_aborts` predicate — **the reward ARITHMETIC is unchanged in both
  branches**; `graph_episode_setup`'s change is passing a `reason=` slug at its three existing
  raise sites and nothing else.
  **WHAT IT IMPLEMENTS, AND WHAT IT DELIBERATELY DOES NOT.** It implements the handoff's
  §3l.6 / §3l.7 harness, population and reporting layer: ONE `episode_design` selector
  resolving the complete approved bundle, the deterministic training cardinality sampler on
  its own SHA-256 rng domain, the 18-stratum matched-triad benchmark MECHANISM (schema,
  builder, canonical serialization, content hash, verifying loader, consumer and identity
  checks), the reason-based `ReferenceIntegrityError` routing decision Task 3 deliberately
  left to Task 4, run-level persistence of every Task-1/2/3 per-episode structure, the derived
  `run_summary.json:/generalized` aggregates, the fourth `measurement_health.png` panel, and
  rollout selector parity. **NOT here:** any final scientific benchmark SCALE, any committed
  or generated benchmark POPULATION, `p(destroy) < 1`, any new `MetaAction`, any reward /
  PPO / GAE / encoder / critic change, and any change to the no-communication boundary. The
  diagnostic rollout gains selector parity and **stays diagnostic** — no matched groups, no
  benchmark, no training.
  **CC-REPORTED ENGINEERING EVIDENCE, LABELLED AS SUCH.** The reviewed tree carries **47**
  new proof tests in `tests/test_graph_generalized.py` (PO1 selector + sampler + rng
  isolation; PO2 the 18 strata, manifest identity, canonical order, refusal and matched-world
  identity, plus the four `test_fix1_*` manifest-integrity regressions and the two
  `test_fix2_*` held-out regressions; PO3 the reference-fault reason split and routing; plus
  the mirror and import-purity guards), and `tests/test_graph_train.py` grew from **123** to
  **144** tests including the `test_gen_*`, `test_fix2_*` and `test_fix3_*` harness,
  benchmark, provenance and persistence regressions.
  `tests/test_graph_reference_continuation.py` stays at **27**, with its Task-3 guard
  `test_po1_no_harness_selects_the_new_policy` deliberately SUPERSEDED by
  `test_po1_the_opt_in_policy_is_reachable_only_through_the_design_selector` — the invariant
  that matters once harness exposure is implemented is the DEFAULT and the ROUTE, not the
  absence. **Those are test COUNTS PRESENT IN THE REVIEWED TREE, not a pass report:** this
  DOCUMENTATION task ran no test suite, no solver, no BLADE episode and no smoke, and it
  makes no pass/fail claim of its own.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #40** — no baseline, no probe,
  no scientific rollout, no generalized campaign, and **no generalized measurement existed at
  this lock, and no generalized result may be pre-claimed** (§8 owns the live run state).
  **PR #40 ITSELF selected no worlds-per-cell scale and committed no benchmark population**
  — a statement about THAT PR's SCOPE, never a current-state claim; **the CURRENT scale /
  authorization state is §8's.** No benchmark manifest is committed or tracked in
  the repository, and transient manifests built by tests and engineering validation are
  neither committed nor a reviewed comparator (this record makes no claim about local
  scratch files); bounded runtime / solver validation was, at THIS lock, a SEPARATE later
  task — it has since been performed and reviewed as Task 5A / Task 5B (§7, below). **No actor-only-vs-CTDE
  generalized result exists.** The approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1
  (`bf1e045f`) measurements are untouched and remain measurements of the `fixed_cell_v1`
  bundle. This lock certifies the IMPLEMENTATION; §8 owns the phase state and the next step.

- `312f586` — **GENERALIZED-V1 TASK 5: the `train_by_*` summary buckets count TRAINING
  attempts only — CLOSED / APPROVED / MERGED.** Approved
  candidate SHA `312f58650b61a85eb72d0554d60715afee862a5c` (committed
  `2026-08-29 21:42:32 +0300`), on branch `task/generalized-v1-task5-summary-phase-fix`,
  whose PARENT is `09eab0673153bd443185ec94530ccf0b042be465` — the `main` head produced by
  the GENERALIZED-V1 Task-4 documentation merge (PR #41) — integrated by merge commit
  `5dfcd8b632be8dca3c1730018bbf35337d07f077` (`2026-08-31 00:06:47 +0300`, **PR #42**),
  Grade A under `GPT_GITHUB`. The candidate was merged with a normal MERGE COMMIT and
  preserved as its SECOND PARENT (ordered parents:
  `09eab0673153bd443185ec94530ccf0b042be465`, then
  `312f58650b61a85eb72d0554d60715afee862a5c`); candidate and integration share the IDENTICAL
  tree `2774effc579ea3f5f2a8cbf5184145985ab68bd6` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. *(SUPERSEDED, and corrected here: this entry previously read "APPROVED,
  NOT YET INTEGRATED at this documentation checkpoint" and stated that no integration SHA
  existed, that `main` was still `09eab067…` and that the eventual merge commit did not yet
  exist. Every one of those was accurate AT THAT CHECKPOINT and is not now — the integration
  SHA is recorded here by the later commit that can name it, exactly as §7's hash convention
  prescribes.)* The technical contract
  is in §5 (the GENERALIZED-V1 TASK 5 block, item 1) and the routing in §6; this entry
  records the LOCK, not the mechanism.
  **THE DEFECT.** `_generalized_summary` built `train_by_agent_count` and
  `train_by_hidden_requested` from the UNFILTERED canonical streams. Both streams mix phases
  by design — an outcome row carries `pre_update` / `train` / `post_update`, a failure row
  carries `train` / `eval` — so held-out EVALUATION attempts were folded into a denominator
  whose NAME says training, and under a frozen benchmark, where every round re-measures the
  same worlds, that bucket's `attempted` would grow with the number of evaluation rounds a
  run happened to perform. The two populations are scheduled independently, so their sum
  describes nothing.
  **THE CORRECTION.** Both streams are filtered to `phase == _ARTIFACT_PHASE_TRAIN` before
  the two named training buckets are built, and `_by` now takes its successful and failed
  populations as EXPLICIT ARGUMENTS instead of closing over them, so the phase a bucket is
  taken over is stated at the call site. Training FAILURES remain represented — the filter is
  on phase, never on outcome — and every other generalized block keeps its own intended
  population.
  **THIS IS A PERSISTED-SUMMARY CORRECTION, NOT AN EPISODE-BEHAVIOUR CHANGE.** No scenario,
  world-construction, reward, solver, PPO, CTDE, fuel-damage, seed, evaluation-schedule or
  population-selection semantics changed; the canonical `episode_outcomes.jsonl` /
  `episode_failures.jsonl` streams are byte-unchanged; and no episode behaves differently.
  **REVIEWED SCOPE: EXACTLY TWO FILES** — `src/match_aou/rl/training/graph_train.py` (43
  changed lines) and `tests/test_graph_train.py`. No other source, test, config, preset,
  README or documentation file was touched.
  **HISTORICAL CC-REPORTED ENGINEERING EVIDENCE ONLY.** The reviewed tree adds the regression
  `test_gen_train_by_buckets_count_training_attempts_only`, taking
  `tests/test_graph_train.py` from **144** to **145** tests. That is a test COUNT PRESENT IN
  THE REVIEWED TREE plus the CC report made at review time — **this DOCUMENTATION task ran no
  test suite, no solver, no BLADE episode and no smoke, and makes no pass/fail claim of its
  own.**
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #42.** This candidate certifies
  a summary-accounting correction; §8 owns the phase state.

- `4af6c5a` — **GENERALIZED-V1 TASK 5: the successful-episode training quota with
  deterministic replacement, and the deterministic benchmark preflight — CLOSED / APPROVED /
  MERGED.** FINAL approved candidate SHA
  `4af6c5aa5dd28072692bfda63282964b55010aae` (committed `2026-08-30 18:02:14 +0300`), on
  branch `task/generalized-v1-task5-success-quota-preflight`, whose PR base was ORIGINALLY
  the PR-#42 branch `task/generalized-v1-task5-summary-phase-fix` and which was RETARGETED to
  `main` once PR #42 was merged — integrated by merge commit
  `b3c2e01f130afe854b09384cd6e1e196de714795` (`2026-08-31 00:13:23 +0300`, **PR #43**),
  Grade A under `GPT_GITHUB`, implementation mode BUILD. **THE RETARGETING CHANGED THE BASE,
  NEVER THE CANDIDATE.** Changing a PR's base invalidates a base-relative verdict even when
  the head SHA does not move, so the unchanged approved head was **EXACT-BASE RE-REVIEWED
  against `main` before merging**, and its effective reviewed delta was byte-identical; the
  head remained `4af6c5aa…` throughout, and the APPEND-ONLY review-fix provenance below is
  untouched by the retarget. The candidate was merged with a normal MERGE COMMIT and
  preserved as its SECOND PARENT (ordered parents:
  `5dfcd8b632be8dca3c1730018bbf35337d07f077`, then
  `4af6c5aa5dd28072692bfda63282964b55010aae`); candidate and integration share the IDENTICAL
  tree `aa5c0cfe456df6e74b0c1fcdb6aed4fa5e1df6d6` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. *(SUPERSEDED, and corrected here: this entry previously read "APPROVED,
  NOT YET INTEGRATED at this documentation checkpoint", stated that no integration SHA
  existed and that none might be invented, and described the candidate as sitting on a
  STACKED base rather than on `main`. All of that was accurate AT THAT CHECKPOINT and is not
  now.)* The technical contract is in
  §5 (the GENERALIZED-V1 TASK 5 block, items 2 through 6) and the routing in §6; this entry
  records the LOCK, not the mechanism.
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. The original implementation candidate
  `734f1e786593b6ffb94f1f8d7283b1f2fc79d257` (committed `2026-08-30 17:05:13 +0300`, parent
  `312f58650b61a85eb72d0554d60715afee862a5c`) carried the quota and the preflight; GPT
  requested ONE review fix, and the correction landed as the DIRECT CHILD COMMIT
  `4af6c5aa…`, which is the APPROVED head and touched TWO files
  (`src/match_aou/rl/training/graph_benchmark_preflight.py`,
  `tests/test_graph_benchmark_preflight.py`).
  **THE REVIEW FINDING AND ITS CLOSURE.** The exhaustion verdict was raised from inside the
  candidate walk, which DISCARDED the cell's candidate outcomes on the way out and left a
  failed preflight with nothing to inspect but an exception message — while the message
  pointed at a build report that had never been written. Every attempted candidate has SPENT
  its seed, so its identity and rejection reason are exactly the evidence an operator needs
  and re-running the same window cannot produce them again differently. The fix SPLITS the
  two concerns: `_scan_cell` now returns its COMPLETE audit and decides nothing, and
  `run_benchmark_preflight` takes the quota verdict afterwards — preserving the exhausted
  cell's audit, scanning no later cell, creating no manifest, building a FAILED report
  through the SAME `_build_report` site the successful path uses, WRITING it before the raise
  when an output directory exists, and attaching it to `BenchmarkPreflightError.report` /
  `.report_path` either way.
  **CUMULATIVE REVIEWED SCOPE: EXACTLY FOUR FILES** —
  `src/match_aou/rl/training/graph_benchmark_preflight.py` (NEW, 1211 lines),
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_benchmark_preflight.py`
  (NEW) and `tests/test_graph_train.py`. **No config, preset or benchmark manifest was added
  or changed** — `configs/graph_train/final_cell_probe.json` remains the ONLY repository
  preset and is untouched and still `fixed_cell_v1` — and no documentation file was part of
  the code candidate, which is what this documentation task closes. No vendored BLADE,
  solver, `graph_reward`, `graph_generalized`, `graph_episode_setup`, PPO, encoder,
  action-space, tick-loop, executor, fuel-damage-mechanism, hidden-placement or generator
  file was touched, and `graph_train.evaluate_benchmark` and `graph_train.evaluate` were NOT
  modified.
  **WHAT IT IMPLEMENTS, AND WHAT IT DELIBERATELY DOES NOT.** It implements the two attempt
  policies and their bounded budget, the run-wide monotone attempt ordinal and the seed
  derived from it, `TrainingQuotaError`, the maximum-possible training seed band every
  held-out claim is made against, and the deterministic per-cell benchmark preflight with its
  complete-manifest rule and durable failed-preflight audit. **NOT here:** any FINAL
  SCIENTIFIC benchmark SCALE or POPULATION, `p(destroy) < 1`, any new `MetaAction`, any
  reward / PPO / GAE / encoder / critic change, any change to post-freeze evaluation, and any
  change to the no-communication boundary.
  **HISTORICAL CC-REPORTED ENGINEERING EVIDENCE ONLY.** The reviewed tree carries the NEW
  `tests/test_graph_benchmark_preflight.py` with **19** PO3 proof tests — the independent
  windows, a rejection unable to shift another cell's accepted seeds, a rejected candidate
  recorded once and never retried, a short realization accepted rather than rejected,
  reproducibility of the same population, a smaller scale as a strict prefix of a larger one,
  window exhaustion aborting with no manifest but a written failed report, an in-memory
  failure carrying its audit without inventing a file, a stale manifest named rather than
  adopted, integrity faults never replacement-eligible, the scale never defaulted, evaluation
  still performing no substitution, and the preflight building no policy and running no
  episode — and takes `tests/test_graph_train.py` from **145** to **159** through the
  fourteen `test_task5c_*` quota, seed-band, provenance and CLI regressions. Those are test
  COUNTS PRESENT IN THE REVIEWED TREE plus the CC report made at review time — **this
  DOCUMENTATION task ran no test suite, no solver, no BLADE episode and no smoke, and makes
  no pass/fail claim of its own.**
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #43** — no baseline, no probe,
  no scientific rollout, no generalized campaign, and **no generalized measurement result
  exists or may be pre-claimed.** **PR #43 ITSELF selected no worlds-per-cell scale and
  committed no benchmark population — it delivered the SELECTION MECHANISM, and the scale
  remains a REQUIRED operator input with no default.** That is a statement about THIS PR's
  SCOPE; **the CURRENT scale / authorization state is §8's, and it records that the R1 scale
  IS selected and its construction IS authorized and dispatched.** No benchmark manifest is
  committed or tracked in the repository, and transient manifests built by tests and
  engineering validation are neither committed nor a reviewed comparator (this record makes
  no claim about local scratch files). The approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1 (`bf1e045f`)
  measurements are untouched and remain measurements of the `fixed_cell_v1` bundle under the
  preserved `scheduled_attempts_v1` policy. This entry certifies the CANDIDATE; §8 owns the
  phase state.

- `bdfd80d` — **GENERALIZED-V1 OPT-IN TRAINING-REWARD EARLY STOPPING
  (`training_reward_plateau_v1`) — CLOSED / APPROVED / MERGED.** Reviewed candidate SHA
  `bdfd80d546e9d5779e4d52b522d5db6d8eb610e9` (committed `2026-09-01 16:49:28 +0300`), on
  branch `task/generalized-v1-early-stopping`, integrated by merge commit
  `0b9a1d63f257a8ed9555f81a1d2bf10e30168e66` (`2026-09-01 18:29:13 +0300`, **PR #48**),
  from base `6f98b4becb39556081389b0e5b48b2dbb7675a5d` — the `main` head produced by the
  PR-#47 post-merge closure merge. Grade A under `GPT_GITHUB`, verdict **APPROVE**. The
  candidate was merged with a normal MERGE COMMIT and preserved as its SECOND PARENT
  (ordered parents: `6f98b4becb39556081389b0e5b48b2dbb7675a5d`, then
  `bdfd80d546e9d5779e4d52b522d5db6d8eb610e9`); candidate and integration share the IDENTICAL
  tree `411126d1d9641356673efbf47510c335b4cf0f9b` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. The reviewed candidate is a SINGLE commit — there was no review-fix
  chain, and none is invented here. The technical contract is in §5 (the GENERALIZED-V1
  early-stopping block) and the routing in §6; this entry records the LOCK, not the
  mechanism.
  **WHAT IT IMPLEMENTS.** ONE OPT-IN stopping policy, `training_reward_plateau_v1`, OFF by
  default and approved for `generalized_v1` only, decided from the persisted
  `train_reward_mean` and from nothing else, identical under `actor_only` and `ctde`, and
  mechanically isolated from every held-out / benchmark / comparator quantity. It adds NO
  episode mechanism, and it changes only WHEN a run may stop consuming its budget.
  **REVIEWED SCOPE: EXACTLY THREE FILES**, verified as the complete
  `6f98b4be…...0b9a1d63…` comparison — `src/match_aou/rl/training/graph_train.py`
  (+616 / −4), `tests/test_graph_train.py` (+647 / −1) and `tests/test_graph_ctde.py`
  (+134 / −0). **No config, preset or benchmark manifest was added or changed**
  (`configs/graph_train/final_cell_probe.json` remains the ONLY repository preset, is
  untouched, is still `fixed_cell_v1`, and does NOT enable early stopping), and **no
  documentation file was part of the code candidate** — that is what this documentation task
  closes. No vendored BLADE, solver, `graph_reward`, `graph_generalized`,
  `graph_benchmark_preflight`, `graph_episode_setup`, `graph_rollout`, PPO, encoder,
  action-space, tick-loop, executor, fuel-damage-mechanism, hidden-placement or generator
  file was touched, and `evaluate`, `evaluate_benchmark` and `save_checkpoint`'s payload were
  NOT modified.
  **WHAT IS DELIBERATELY NOT IN IT.** No loader and no resume semantics — checkpoints stay
  SAVE-only and restoring a run remains DEFERRED. No change to the PLANNED
  `max_training_attempts` or to any held-out / seed-band claim made against it. No scenario,
  world-construction, reward, solver, PPO, CTDE, fuel-damage, seed-formula, episode-design,
  cardinality-sampler, manifest, preflight or evaluation-schedule change. `p(destroy)` stays
  `1.0`. **No new `MetaAction`**, and nothing from this layer reaches the acting path.
  **HISTORICAL CC-REPORTED ENGINEERING EVIDENCE ONLY.** The reviewed tree carries the ten
  `test_es_*` proof tests the packet enumerates plus two configuration-surface tests, taking
  `tests/test_graph_train.py` from **159** to **171** tests, and the actor-only-vs-CTDE
  parity test `test_early_stopping_is_identical_under_actor_only_and_ctde`, taking
  `tests/test_graph_ctde.py` from **45** to **46**. Those are test COUNTS PRESENT IN THE
  REVIEWED TREE plus the CC report made at review time — **this DOCUMENTATION task ran no
  test suite, no solver, no BLADE episode and no smoke, and makes no pass/fail claim of its
  own.**
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #48**, and none may be
  inferred from it. **No scientific run has used this mechanism**, **no reward, convergence,
  runtime-saving or performance claim is made or supported for it**, and **firing the rule
  would record only that the configured training-reward plateau rule fired — never a
  convergence or optimality claim.** The dispatched actor-only R1 is UNTOUCHED by this work
  and remains governed by its own FIXED-BUDGET contract with NO early stopping; it stays
  `AUTHORIZED / DISPATCHED — RESULT PENDING` with nothing about its outcome stated or
  inferable. The approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1 (`bf1e045f`)
  measurements are untouched and remain measurements of the `fixed_cell_v1` bundle under the
  preserved fixed-budget path. This entry certifies the IMPLEMENTATION; §8 owns the phase
  state.

- `8f0d250` — **MATCH-AOU DETERMINISTIC-`p=1` SOLVER + EXPLICIT BACKEND INTEGRATION —
  CLOSED / APPROVED / MERGED.** FINAL approved candidate SHA
  `8f0d250cd9f96e6b8bce635065701dc47a5ee87e`, integrated by merge commit
  `9979910a0537e829f1d18483011e4d0fab42c257` (**PR #54**), from base
  `fd0d668d5031adef1f3b6af612e584f9ab56454b` — the `main` head produced by the R1-review
  documentation merge (PR #53). The candidate was merged with a NORMAL MERGE COMMIT and
  preserved as its SECOND PARENT (ordered parents:
  `fd0d668d5031adef1f3b6af612e584f9ab56454b`, then
  `8f0d250cd9f96e6b8bce635065701dc47a5ee87e`); candidate and integration share the IDENTICAL
  tree `9507dc0bc16aeeabf5616171e10f5a28480063ec` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. Grade A under `GPT_GITHUB`. The technical contract is in §5 (the
  MATCH-AOU allocation-backend block) and the routing in §6; this entry records the LOCK,
  not the mechanism.
  **THE APPROVED ISOLATED-SOLVER ANCESTOR is `1462163277322a3ef29eec28c782766edb8ea73b`** —
  the reviewed stage at which the deterministic P1 MILP existed as an ISOLATED module with
  no runtime caller. The integration candidate above is what added the reviewed SEAM beside
  it; the two stages are recorded separately because the approved P1 FORMULATION and its
  INTEGRATION are different decisions.
  **WHAT IT IMPLEMENTS.** ONE explicit, INDEPENDENT selector over exactly two
  non-interchangeable objectives — `legacy_minlp_v1` (the historical DEFAULT: the frozen
  MINLP through BONMIN) and `p1_milp_v1` (the deterministic `p = 1` MILP through
  SciPy/HiGHS, with no `EPSILON`). No `auto`, no fallback in either direction, no per-solve
  switching, one backend per episode stored on `EpisodeContext.match_aou_backend` and read
  back by every deferred reference solve, objective-coherent reward valuation, and
  `MatchAouBackendError` routed as a configuration/instrument ABORT. §5 states all of it.
  **REVIEWED SCOPE: EXACTLY ELEVEN FILES**, verified as the complete
  `fd0d668…...9979910…` comparison — `src/match_aou/solvers/match_aou_backend.py` (NEW),
  `src/match_aou/solvers/match_aou_p1_milp_solver.py` (NEW),
  `src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_reward.py`,
  `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_rollout.py`,
  `src/match_aou/rl/training/graph_benchmark_preflight.py`,
  `tools/benchmark_match_aou_p1_milp.py` (NEW),
  `tests/test_match_aou_backend_integration.py` (NEW),
  `tests/test_match_aou_p1_milp_solver.py` (NEW) and `tests/test_graph_setup_seam.py`.
  **The FROZEN `match_aou_MINLP_solver.py` was NOT touched**, `match_aou.solvers.__init__`
  was NOT touched (its surface is still exactly `MatchAou` and `round_trip_cost`), the
  vendored BLADE engine was NOT touched, `graph_generalized.py` was NOT touched (**no
  benchmark-manifest schema change**), and **no config, preset or benchmark manifest was
  added or changed** — `configs/graph_train/final_cell_probe.json` remains the ONLY
  repository preset, is untouched, is still `fixed_cell_v1` and does NOT select
  `p1_milp_v1`. No documentation file was part of the code integration, which is what this
  documentation task closes.
  **WHAT IS DELIBERATELY NOT IN IT.** No claim of solver equivalence and no claim of literal
  one-config-field experimental equivalence between a legacy arm and a P1 arm; no `p < 1`,
  multi-step or precedence support in the P1 formulation (those are REFUSED, not answered);
  no tie-breaking rule and no degeneracy-breaking bound; no change to `U_prefix`, `U_post`,
  `realized_utility`, the aircraft penalty, `eps_regret`, terminal-on-last credit placement
  or the no-clamping policy; no PPO, GAE, encoder, critic, action-space, trigger,
  fuel-damage-mechanism, `DETECTION_KM`, B2-geometry or seed-formula change; and no
  episode-outcome schema bump (it stays at version 3).
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS PRODUCED BY THIS INTEGRATION**, and none may
  be inferred from it: its tests and its `tools/benchmark_match_aou_p1_milp.py` comparison
  are ENGINEERING evidence with no scientific contract, no seed schedule, no held-out band
  and no denominator, and **no P1 performance, benefit, learning or comparison claim may be
  pre-claimed** (§8). The approved Phase-A (`737b4bf`), FD-VARIABLE-SEVERITY-v1
  (`bf1e045f`) and R1 (`4af6c5aa…`) measurements are untouched and remain measurements
  taken under `legacy_minlp_v1`. This entry certifies the IMPLEMENTATION; §8 owns the phase
  state.

- `d36e133` — **CERTIFIED-FD PHYSICAL-STATE INTEGRITY REPAIR — CLOSED / APPROVED /
  MERGED.** FINAL approved candidate SHA
  `d36e1338aaac0d55dd081b788a3e8bbcaa310b53`, integrated by merge commit
  `edf9e840a30a4a4c3b2ef6daa319661c1d6f3cb8` (**PR #55**), from base
  `9979910a0537e829f1d18483011e4d0fab42c257` (the PR-#54 merge). The candidate was merged
  with a NORMAL MERGE COMMIT and preserved as its SECOND PARENT (ordered parents:
  `9979910a0537e829f1d18483011e4d0fab42c257`, then
  `d36e1338aaac0d55dd081b788a3e8bbcaa310b53`); candidate and integration share the IDENTICAL
  tree `0e3c0ff8bc41e5d1d96af9ec3d61a4b5cea59afa` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. Grade A under `GPT_GITHUB`. The technical contract is in §5 (the live
  certificate-check block) and the routing in §6; this entry records the LOCK, not the
  mechanism.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. The original implementation candidate
  `930987c7bdc19596383a4c4b825f064817812375` carried the FD repair; GPT returned
  **REQUEST FIXES**, and the correction landed as the DIRECT CHILD COMMIT `d36e1338…`, which
  is the APPROVED head. **THE REQUESTED FIX CONCERNED THE P1 HISTORICAL-SURFACE TEST, NOT FD
  PRODUCTION SEMANTICS**: the FD runtime correction was accepted in the first candidate.
  **THE REVIEW-FIX CHILD DELTA AND THE CUMULATIVE PR SURFACE ARE DIFFERENT QUANTITIES AND
  MUST NOT BE CONFLATED.** The CHILD DELTA `930987c7…` → `d36e1338…` changed **EXACTLY ONE
  FILE — `tests/test_match_aou_p1_milp_solver.py`** — and both
  `src/match_aou/rl/training/graph_fuel_damage.py` and `tests/test_graph_fuel_damage.py`
  were **BYTE-IDENTICAL** across that child commit, so no production FD behaviour and no FD
  regression was re-decided by the fix. The CUMULATIVE PR #55 / integrated surface is the
  THREE files recorded below.
  **THE DEFECT.** `FuelDamageController._require_certificate_holds` bound the ABSOLUTE OUTER
  TICK, on the premise that an airborne ego receives exactly one engine update per outer
  tick. **THE PREMISE, NOT THE CERTIFIER, WAS WRONG**: frozen BLADE's
  `Game.update_all_aircraft_position` iterates the live `scenario.aircraft` list while
  `land_aicraft` → `remove_aircraft` and the fuel-exhaustion branch remove entries from that
  same list, so the entry following a departing aircraft can be skipped entirely for that
  update — losing BOTH its movement and its burn (§2). An ego whose peers land is therefore
  physically EARLIER than the tick count implies.
  **THE REPAIR.** Setup-time certification stays TICK-AWARE and byte-unchanged
  (`event_tick`, `movement_count`, `bracket_ticks`, `CERTIFICATE_TICK_TOLERANCE == 1` and
  the tolerance derivations built from that quantum), while LIVE validation binds ONLY the
  ego's PHYSICAL state — position against the certificate's existing `position_tolerance_km`
  and pre-damage fuel against its existing `fuel_tolerance`. **NEITHER TOLERANCE WAS
  WIDENED, none was made dynamic, and NO engine-update counter was added**; all three deltas
  are computed before any verdict and reported together; a genuine physical contradiction
  still raises `FuelDamageIntegrityError` BEFORE the fuel mutation; and world acceptance,
  certificate construction, the terminal certified-damaged-event-never-realized integrity
  abort and the ordinary `NO_FD_ELIGIBLE_EGO` setup attrition are all unchanged. **BLADE WAS
  DELIBERATELY NOT MODIFIED — this is NOT a physics fix and must never be read as one.**
  **REVIEWED SCOPE: EXACTLY THREE FILES**, verified as the complete
  `9979910…...edf9e840…` comparison — `src/match_aou/rl/training/graph_fuel_damage.py`,
  `tests/test_graph_fuel_damage.py` and `tests/test_match_aou_p1_milp_solver.py`. No
  vendored BLADE, solver, reward, PPO, encoder, action-space, tick-loop, executor,
  episode-setup, hidden-placement, generator, trainer, rollout, preflight, config, preset,
  manifest or documentation file was touched.
  **THE DURABLE P1 HISTORICAL-SURFACE REGRESSION.**
  `tests/test_match_aou_p1_milp_solver.py::test_po2_the_reviewed_p1_task_modified_only_its_declared_surface`
  now compares the TWO PINNED HISTORICAL COMMITS — `TASK_BASE_SHA =
  "fd0d668d5031adef1f3b6af612e584f9ab56454b"` against `P1_REVIEWED_SHA =
  "8f0d250cd9f96e6b8bce635065701dc47a5ee87e"` — and **NOT the current `HEAD`**. That is what
  preserves the PR-#54 surface proof as the finished historical fact it is WITHOUT
  prohibiting future repository evolution: diffing the pinned base against the working tree
  was a valid claim only while PR #54 was itself the current task, and afterwards any
  legitimate later change by any task would have failed a test whose name promises something
  about P1. It is NON-VACUOUS by construction (the change set must be non-empty and must
  EQUAL the declared inventory in both directions) and remains FALSIFIABLE through
  `test_po2_the_historical_surface_guard_still_rejects_an_undeclared_file`, while the LIVE
  tree stays guarded by the byte-for-byte frozen-MINLP pin and the P1 AST guard.
  **CC-REPORTED ENGINEERING EVIDENCE ONLY.** At the approved candidate the reported full
  suite was **659 passed, 11 skipped, 0 failed**, and a bounded **seed-740322** reconstruction
  / replay reproduced the diagnosed skipped-update signature. **BOTH ARE ENGINEERING
  VALIDATION, NOT A SCIENTIFIC MEASUREMENT** — the replay schedules no population, defines
  no comparator and produces no verdict, and no reward, learning or performance claim may be
  drawn from it. **This DOCUMENTATION task ran no test suite, no solver, no BLADE episode
  and no smoke, and makes no pass/fail claim of its own.**
  **NO SCIENTIFIC P1 RUN WAS LAUNCHED OR RESUMED BY PR #55**, and no scientific measurement
  of any kind was produced by it. The approved Phase-A (`737b4bf`),
  FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) and R1 (`4af6c5aa…`) measurements are untouched.
  This entry certifies the IMPLEMENTATION; §8 owns the phase state, including the ABORTED
  P1 arm that must not be resumed.

- `a27a3b1` — **GENERALIZED-V2: THE TWO-STAGE ROUTE-RELATIVE POPULATION — CLOSED /
  APPROVED / MERGED.** FINAL approved candidate SHA
  `a27a3b140248e95096db38c8d5f717cc0098da4d`, integrated by merge commit
  `f98b293ededf7f67fbbd8f742e797e08109c254b` (**PR #57**), from base
  `ae1941035991df4719df212c4b5dd07db89aee4a` (the PR-#56 merge). The candidate was merged
  with a NORMAL MERGE COMMIT and preserved as its SECOND PARENT (ordered parents:
  `ae1941035991df4719df212c4b5dd07db89aee4a`, then
  `a27a3b140248e95096db38c8d5f717cc0098da4d`); candidate and integration share the IDENTICAL
  tree `32bc0460cfa1fa2a445dc7b0081a8f085f2c7814` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. Grade A under `GPT_GITHUB`. The technical contract is in §5 (the
  GENERALIZED-V2 block), the selector's pipeline placement in §4, and the routing in §6.
  This entry records the LOCK, not the mechanism.
  **APPEND-ONLY REVIEW CHAIN — FOUR COMMITS ON ONE BRANCH AND ONE PR**, never amend,
  rebase, squash, force-push or history rewrite. The original implementation candidate
  `3a6653a` (`feat(generalized-v2): route-relative hidden-load population, resolved after
  the solve`) carried the layer, and three review-fix commits landed as DIRECT CHILDREN on
  the same branch `task/generalized-v2-route-relative-population` — `aa9677e` (preserve the
  EXACT stage-2 population on a failed attempt), then `7a7f41e` (the backend selector is
  EXPLICIT but DESIGN-CONSTRAINED), then the APPROVED head `a27a3b1` (complete the CURRENT
  three-design operator contract).
  **WHAT IT IMPLEMENTS.** A THIRD episode-design bundle, `generalized_v2`, resolving the
  IDENTICAL four low-level policy ids as `generalized_v1` and changing only the POPULATION:
  a TWO-STAGE, route-relative cardinality — `A ~ U{2,3,4,5,6}` and `K | A ~ U{A, A+2}`
  BEFORE the known-only solve, then `H_requested ~ U{1..R}` AFTER it, where `R` is the
  number of egos that solve actually routed — on two disjoint SHA-256 seed domains of its
  own; the `p1_milp_v1` backend requirement, refused before execution; the write-once,
  caller-owned `RouteRelativePopulationRecorder` and the stage-aware failure provenance
  built on it; and rollout selector parity. §5 states all of it.
  **REVIEWED SCOPE: EXACTLY TEN FILES**, verified as the complete `ae194103…...f98b293e…`
  comparison — `src/match_aou/rl/training/graph_generalized.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_hidden_placement.py`,
  `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_rollout.py`,
  `src/match_aou/rl/training/graph_benchmark_preflight.py`,
  `tests/test_graph_generalized_v2.py` (NEW), `tests/test_graph_generalized.py`,
  `tests/test_graph_setup_seam.py` and `tests/test_graph_train.py`. **No config, preset or
  benchmark manifest was added or changed** — `configs/graph_train/final_cell_probe.json`
  remains the ONLY repository preset, is untouched and is still `fixed_cell_v1` — and **no
  documentation file was part of the code integration**, which is what this documentation
  task closes. The frozen `match_aou_MINLP_solver.py`, the P1 MILP solver, the backend
  module, the vendored BLADE engine, `graph_reward`, `graph_ppo`, `graph_encoder`,
  `graph_action`, `graph_effect`, `graph_trigger`, `graph_tick_loop`, `graph_fuel_damage`,
  the executor, the generator and `central_graph_builder` were all UNTOUCHED. The
  `graph_benchmark_preflight` change is a NINE-LINE refusal tightening
  (`cfg.generalized` → `cfg.design.generalized_v1_design`) and nothing else.
  **WHAT IS DELIBERATELY NOT IN IT.** No GENERALIZED-V2 evaluation construct, benchmark,
  stratification, LOW/HIGH interpretation, matched-group construction, worlds-per-cell
  scale, seed band or manifest identity — `validate()` REFUSES a `benchmark_manifest` and
  REFUSES evaluation outright under V2, `evaluate()` and `evaluate_benchmark()` RAISE, and
  the preflight refuses anything that is not EXACTLY `generalized_v1`. No early-stopping
  extension (the approved plateau policy stays `generalized_v1`-only). No
  multi-hidden-per-route placement and no placement-geometry change. `p(destroy)` stays
  `1.0`. **No new `MetaAction`**, no observation/action/mask/reward/PPO/GAE/CTDE change, and
  no change to the no-communication boundary. The episode-outcome schema stays at
  **version 3** — the V2 population block is added conditionally, not by a schema bump.
  **HISTORICAL CC-REPORTED ENGINEERING EVIDENCE ONLY.** The reviewed tree adds the NEW
  `tests/test_graph_generalized_v2.py` with its `test_po1_*` preservation, `test_po2_*`
  two-stage / backend / operator-surface and `test_po1_the_failure_population_*` provenance
  proofs, and extends `tests/test_graph_setup_seam.py` with the five `test_v2fix_*`
  recorder regressions and `tests/test_graph_train.py` with the four `test_v2fix_*` ledger
  regressions. Those are test FILES AND NAMES PRESENT IN THE REVIEWED TREE plus the CC
  report made at review time — **this DOCUMENTATION task ran no test suite, no solver, no
  BLADE episode and no smoke, and makes no pass/fail claim of its own.**
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS PRODUCED BY PR #57**, and none may be
  inferred from it: **it produced NO scientific training measurement, NO V2 benchmark and
  NO V2 policy-performance result**, and **the Grade-A IMPLEMENTATION grade must not be
  projected onto any future V2 measurement.** The approved Phase-A (`737b4bf`),
  FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) and R1 (`4af6c5aa…`) measurements are untouched and
  remain measurements of the designs they were taken on. This entry certifies the
  IMPLEMENTATION; §8 owns the phase state.

- `786e821` — **GENERALIZED-V2 BENCHMARK: THE FROZEN TEN-CELL BENCHMARK AND EVALUATION
  CONSTRUCT — CLOSED / APPROVED / MERGED.** FINAL approved candidate SHA
  `786e8218a00954f7a7f20fe1dfca93ec71a400d4`, integrated by merge commit
  `ea8778d5010fcfccec357c57c2861606ecb58bbc` (**PR #59**), from original base
  `fe7b449c94281bb12fdadc12be89ee36f447c79a` — the PR-#58 GENERALIZED-V2 documentation-lock
  merge, whose SHA this later entry may now name under the hash convention. The candidate was
  merged with a NORMAL MERGE COMMIT and preserved as its SECOND PARENT (ordered parents:
  `fe7b449c94281bb12fdadc12be89ee36f447c79a`, then
  `786e8218a00954f7a7f20fe1dfca93ec71a400d4`); candidate and integration share the IDENTICAL
  tree `727a65f072ae46d72f61287b5c43dbfda9adf72d` (verified locally), so the integrated tree
  is exactly the reviewed tree, and no rebase, squash, cherry-pick, force-push or history
  rewrite occurred. Grade A under `GPT_GITHUB`. The technical contract is in §5 (the
  GENERALIZED-V2 BENCHMARK block) and the routing in §6; this entry records the LOCK, not the
  mechanism.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. The first candidate
  `735e1fd89c230590b75fcaf506a7f470c92b2dc9` (parent `fe7b449c…`) carried the construct and
  received **REQUEST FIXES**; the correction landed as the DIRECT CHILD COMMIT `786e8218…`,
  which is the APPROVED head, touching SEVEN files across three named fixes: (1)
  FAIL-CLOSED bounded-backoff exhaustion — the V2 preflight had treated ANY
  `HiddenPlacementError` as replacement-eligible, so it now recognizes only the new
  `BoundedBackoffExhaustedError` subtype and a plain `HiddenPlacementError` aborts; (2) V2
  manifest SEMANTIC validation — `V2WorldPreflight` now refuses any frozen population state
  production V2 could not produce, so a self-consistently re-hashed impossible manifest is
  refused; (3) `evaluate()`'s V2 refusal WORDING, with no control-flow change. *(The review
  verdicts are the orchestrator's review records; PR #59 carries no GitHub review objects.)*
  **CUMULATIVE REVIEWED SCOPE: EXACTLY NINE FILES**, verified as the complete
  `fe7b449c…...ea8778d5…` comparison —
  `src/match_aou/rl/training/graph_generalized.py`,
  `src/match_aou/rl/training/graph_benchmark_preflight.py`,
  `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_hidden_placement.py`,
  `tests/test_graph_generalized_v2_benchmark.py` (NEW), `tests/test_graph_generalized_v2.py`,
  `tests/test_graph_hidden_placement.py` and `tests/test_graph_setup_seam.py`. **TWO
  AUTHORIZED SCOPE EXTENSIONS, recorded rather than hidden:** `graph_episode_setup.py` was
  added to scope DURING IMPLEMENTATION, with the orchestrator's authorization, solely to
  classify the route-relative `R == 0` refusal as the typed `RouteRelativeNoRoutesError`;
  and the REVIEW-FIX commit was separately authorized to make a CLASSIFICATION-ONLY change to
  the LOCKED B2 layer `graph_hidden_placement.py` (`BoundedBackoffExhaustedError` at the one
  existing terminal zero-realized branch — geometry, candidate order, RNG, acceptance and
  historical `HiddenPlacementError` compatibility unchanged). **No config, preset, benchmark
  manifest or documentation file was part of the code integration** —
  `configs/graph_train/final_cell_probe.json` remains the ONLY repository preset and is still
  `fixed_cell_v1` — and `graph_rollout`, `graph_reward`, `graph_ppo`, `graph_encoder`,
  `graph_action`, `graph_effect`, `graph_trigger`, `graph_tick_loop`, `graph_fuel_damage`,
  the executor, the generator, both solvers, the backend module and vendored BLADE were all
  UNTOUCHED.
  **HISTORICAL CC-REPORTED ENGINEERING EVIDENCE ONLY.** At the first candidate CC reported a
  full base-env suite of **735 passed, 11 skipped**, `nlp_env` runners train **175/175**,
  preflight **19/19**, generalized_v2 **31/31**, v2_benchmark **33/33** and setup_seam
  **59/59, 0 skipped** (real BLADE + BONMIN tiers), plus a two-cell ENGINEERING construction
  smoke writing no manifest; at the approved head it reported **738 passed, 11 skipped**,
  hidden_placement **33/33**, v2_benchmark **36/36**, generalized_v2 **31/31**, preflight
  **19/19**, train **175/175**, an unchanged V1 manifest golden and a clean `git diff
  --check`. The reviewed tree carries **36** test functions in the NEW
  `tests/test_graph_generalized_v2_benchmark.py`. **This DOCUMENTATION task ran no test
  suite, no solver, no BLADE episode and no smoke, and makes no pass/fail claim of its own.**
  **NO SCIENTIFIC ARTIFACT OF ANY KIND WAS PRODUCED BY PR #59** — no scientific benchmark
  manifest, no real benchmark seed selection, no preflight invocation, no training run and no
  scientific V2 measurement — and **its Grade-A IMPLEMENTATION approval must NOT be projected
  onto any future scientific measurement.** The approved Phase-A (`737b4bf`),
  FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) and R1 (`4af6c5aa…`) measurements, and the fresh
  deterministic-P1 arm measured at `ae194103…`, are untouched. This entry certifies the
  IMPLEMENTATION; §8 owns the phase state.

---

## 3. Merged pull requests

Derived from `git log --merges --first-parent main` at `ae42cb01677f94868b2873008d87be677e31f0c8`.
"Merged head" is the merge commit's second parent. Review verdicts are recorded in the entries
above, not in this table.

| PR | Branch | Merged head | Merge commit | Date |
|---|---|---|---|---|
| #2 | `task/b1-generator-configuration` | `d6758ac1899621b2ceebcb63afb5e8577184cd91` | `bd087c3c18b96f1fe847b4987c73f394a43249c1` | 2026-07-28 |
| #3 | `task/b2-route-relative-hidden-placement` | `e22aee359e06591bdb179ef06a566db90f83a558` | `8db9428147b77e9432e7ad6b085dc5898c9062bb` | 2026-07-28 |
| #4 | `task/b3-setup-seam` | `dd14ab418c71e3bd615f1198d0c612502642d29b` | `14224531db9deb700f6e397203177eb8c701c6cc` | 2026-07-29 |
| #5 | `task/b3-documentation-lock` | `f991a5c29a78e4880179426b436550f17ff47529` | `ccccd8d715be3e2b44a4204fdca26a5ec4cb0e8e` | 2026-07-29 |
| #6 | `task/b4-training-run-instrumentation` | `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c` | `ba936606deada050ed9298600ee9041fc330af6c` | 2026-07-29 |
| #7 | `task/b4-per-episode-observability` | `211e12e49b676637362d42effdb80988dd0e55eb` | `ffb95a6ee90df45b2d89802b321dcadcbc272821` | 2026-08-01 |
| #8 | `task/fuel-damage-baseline-v1` | `a8669f450708c2508753c49ab16fd1028b29607d` | `1cecb0ac99f839d47ffeea12c8871aec77e66640` | 2026-08-03 |
| #9 | `task/fd-baseline-v1-doc-lock` | `1a0267726bebe436c218f7c1997a963cc274b4ab` | `440991e519ee29a95e9682164f35de9b41f87a6a` | 2026-08-03 |
| #10 | `task/final-cell-visual-artifacts` | `24d1835f31d2e6aac04b418308a8753c392ac951` | `771f2107211fb3f984b64482b799613260e19aca` | 2026-08-11 |
| #11 | `task/repo-code-hygiene` | `2a3f89cf2d027581308493a98767ae658107d6d1` | `6e2757dd30100f429d492f4d23fd8b5f57cf4fac` | 2026-08-13 |
| #12 | `task/repo-doc-hygiene` | `52064c2d306df7c8447d159df20e6e189a59bf85` | `5f78904e3af1e2e47386c9b0e01ddbaa273724f5` | 2026-08-14 |
| #13 | `task/repo-hygiene-closure-docs` | `0f8c1c248040ea8b971d7183d4fc7bb62658070c` | `ffcf6d16a43e0b464e55de4c7e17d7d9aff93632` | 2026-08-14 |
| #14 | `task/final-cell-probe-config-and-plots` | `61e539ed62fcf1e3fe25a83d213cae06f5afa98e` | `a5f389a2af328640e19db51d3277a33167c08f25` | 2026-08-14 |
| #15 | `task/lock-final-cell-probe-harness` | `b8f1332666c38f4bb60d73b1770c533866570feb` | `238062d7d284334432d9c39d7543fb0bbf39ea7c` | 2026-08-14 |
| #16 | `task/record-short-probe-validity-findings` | `463bf181e62c9c8cc463f64cc8aa281e10823656` | `df5bdf9f4317974d0a3f247d26fe37d2906d51b5` | 2026-08-15 |
| #17 | `task/defect-a-ego-global-spa` | `d56fda636ab5ec1a5cce6076f07acac5556d10cb` | `f094e0b32e5e67b79757edbfe4e73c1fe01b0a87` | 2026-08-16 |
| #18 | `task/defect-a-doc-lock` | `ae632934b5c272b16613ea8679c2a707042e8749` | `cefda78b18ea2daeda5014bab9a75a0945ef8e37` | 2026-08-16 |
| #19 | `task/defect-b-derived-confirmation-wait` | `39a16f2e5e1a3302d545c11b072e037e9702dffe` | `60a82d17398e9d14be1c2684cc72fafd020e0d9b` | 2026-08-16 |
| #20 | `task/defect-b-doc-lock` | `b41989bbfff1dafea7af053dd0ea185c08ed2153` | `6e97940733d2c7cf8c4ffc7033180c65f644ae17` | 2026-08-16 |
| #21 | `task/defect-c-physical-rtb-completion` | `ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b` | `0de9f21eb9e8904f06f836f4ecd010bc46c788b6` | 2026-08-16 |
| #22 | `task/defect-c-doc-lock` | `8858d1457d462c4c943dd539814845e355fc2681` | `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` | 2026-08-16 |
| #23 | `task/corrected-short-probe-doc-lock` | `0ecd836eac7a65f509a74c10158ca8a36a3f26c2` | `c30b6982ba605d60976cc303256da4b5528b0e63` | 2026-08-16 |
| #24 | `task/roster-world-truth-fix` | `36365f210e8a659a641a7713f612c7e0ec1d4665` | `f37ea1c8559405d5de24a9c2dd9e740227acaeeb` | 2026-08-17 |
| #25 | (merge titled "record roster-integrity correction and invalidate affected runs") | `6a62168bff1857890dda14eed3766444fb5c68e6` | `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` | 2026-08-17 |
| #26 | (merge titled "record the VALID Phase-A long baseline and hand off to Phase-B CTDE") | `2d0dade2c69f42d4889d851cf8ecc5fbd0ccfc30` | `4f0068847b017795717c5f0e331f647bcfc30547` | 2026-08-19 |
| #27 | `task/variable-fd-severity-baseline` | `eecc9b5d91bce4a98a070a29307cc12af0d4c4a3` | `177e969446ef6c01c729484f2ea9969c94a27330` | 2026-08-20 |
| #28 | `task/variable-fd-severity-doc-lock` | `0db6cebbb956ec7d05a064086c4b13cd7105ea57` | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | 2026-08-20 |
| #29 | `task/ctde-parallel-order-doc-lock` | `1aa8eef865351959d61e229de86040620cb2cb50` | `9018d74a09c59eec1a3c7ce1280f7518036e72ee` | 2026-08-22 |
| #31 | `task/fd-variable-severity-valid-doc-lock` | `92941587b1ad225573af763e50b129a552861b18` | `d437084c5fb1a22c21596a48c58e03f7e15a0115` | 2026-08-23 |
| #30 | `task/phase-b-ctde-build` | `a6f3aa9d62931994f416b2241fec4cfac3b018ec` | `8390d85c2072e9cbe984ce5f2731cef3a9b14985` | 2026-08-23 |
| #32 | `task/phase-b-ctde-doc-lock` | `c607f3fabcbd58f6f10cfde6bcc34068f09e4121` | `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96` | 2026-08-23 |
| #33 | `task/ctde-chat-closure-handoff` | `802696b4adf702ef78aa459d470d3f24cb76cc49` | `76abdc480e80a84f1503208730d4525cd5e89b69` | 2026-08-23 |
| #34 | `task/generalized-v1-handoff-bootstrap` | `878210009a772977e54d589a9100a86e92b1ff2e` | `7b86098a7573be15b0d8bfcf959b1d1f63288ffc` | 2026-08-25 |
| #35 | `task/generalized-v1-cardinality-b2` | `5b55ca348309b4241d2087c2f60327bc842ea6fa` | `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9` | 2026-08-25 |
| #36 | `task/generalized-v1-fd-adaptation` | `185d39f00335a0bb5e9130cc773da94c914f17f5` | `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b` | 2026-08-25 |
| #37 | `task/generalized-v1-task12-doc-lock` | `a43fc7bff564ca32a0152c7592b12d1f54ef732b` | `ca2fe346b5fb5d499b6a59b3da17c74b2a8bae8e` | 2026-08-25 |
| #38 | `task/generalized-v1-task3-continuation-reference` | `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e` | `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25` | 2026-08-25 |
| #39 | `task/generalized-v1-task3-doc-lock` | `2c2bf6479bcd32c29338b8b637484b0262d60e02` | `f4e8d3b8ddc61525fe0cde6b61ca4d611ebd2eed` | 2026-08-26 |
| #40 | `task/generalized-v1-task4-harness-benchmark` | `db79013897a6e5669f50d53b6e30229b16aea28d` | `b4daa8c1a8c870061b26cceb01d4ed34169594e7` | 2026-08-26 |
| #41 | `task/generalized-v1-task4-doc-lock` | `e140888d3bc6b36b2ab15359d1f10325a072bb60` | `09eab0673153bd443185ec94530ccf0b042be465` | 2026-08-26 |
| #42 | `task/generalized-v1-task5-summary-phase-fix` | `312f58650b61a85eb72d0554d60715afee862a5c` | `5dfcd8b632be8dca3c1730018bbf35337d07f077` | 2026-08-31 |
| #43 | `task/generalized-v1-task5-success-quota-preflight` | `4af6c5aa5dd28072692bfda63282964b55010aae` | `b3c2e01f130afe854b09384cd6e1e196de714795` | 2026-08-31 |
| #44 | `task/generalized-v1-task5-doc-lock` | `88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` | `9b9e9b85a70c8a0019c72ada92ceec3401725795` | 2026-08-31 |
| #45 | `task/generalized-v1-task5-post-integration-closure` | `728ebf3f4070ec999baef8d3aacc364b7e2a2776` | `926aba66fcaf2b99fc58685eb202888d8deeaf5f` | 2026-08-31 |
| #46 | `task/cluster-env-repro-lock` | `cbc227450067d96c630eed208e22b3a5a20efc1b` | `e9f9f4f93412c8c6c3dd8ba81a7e784dc52cc68b` | 2026-08-31 |
| #47 | `task/cluster-env-post-merge-closure` | `0e1be782da5d367f358df9800d6db3328b3d52d5` | `6f98b4becb39556081389b0e5b48b2dbb7675a5d` | 2026-08-31 |
| #48 | `task/generalized-v1-early-stopping` | `bdfd80d546e9d5779e4d52b522d5db6d8eb610e9` | `0b9a1d63f257a8ed9555f81a1d2bf10e30168e66` | 2026-09-01 |
| #49 | `task/generalized-v1-early-stopping-doc-lock` | `77c26dde1396acc7793d50fbcac840474601bf88` | `f74c288175a1f8228407806bf5c8056beff75239` | 2026-09-02 |
| #50 | `task/generalized-v1-early-stopping-post-merge-closure` | `a7d6dea5375a809e8b59aaee19f763f5769499ea` | `e9cbd80244926680d90c81d9440753b89e22efdc` | 2026-09-02 |
| #51 | `task/generalized-v1-early-stopping-final-handoff-stabilization` | `39004fb799e67481fe64f89bca7320fde1dcced6` | `44530abb1cc3f99d01ac867c6621047ac9343661` | 2026-09-02 |
| #52 | `task/generalized-v1-fd-measurement-hardening` | `81a148f80317499d8897db44bd713976962db832` | `28eb8dad2643fc79d516b47ec95119a395e76257` | 2026-09-05 |
| #53 | `task/generalized-v1-r1-review-doc-lock` | `ee8fd0043eaed490c89e7e6356ae590c7c4bc65e` | `fd0d668d5031adef1f3b6af612e584f9ab56454b` | 2026-09-05 |
| #54 | `task/match-aou-p1-milp-solver` | `8f0d250cd9f96e6b8bce635065701dc47a5ee87e` | `9979910a0537e829f1d18483011e4d0fab42c257` | 2026-09-06 |
| #55 | `task/fd-certificate-physical-state-integrity` | `d36e1338aaac0d55dd081b788a3e8bbcaa310b53` | `edf9e840a30a4a4c3b2ef6daa319661c1d6f3cb8` | 2026-09-06 |
| #56 | `task/p1-fd-post-integration-doc-lock` | `d295561ba2d4f459118eca7d13d8d266dfd66fc4` | `ae1941035991df4719df212c4b5dd07db89aee4a` | 2026-09-06 |
| #57 | `task/generalized-v2-route-relative-population` | `a27a3b140248e95096db38c8d5f717cc0098da4d` | `f98b293ededf7f67fbbd8f742e797e08109c254b` | 2026-09-12 |
| #58 | `task/generalized-v2-doc-lock` | `773defa2cebe87bec7e862c7adf27b5059941485` | `fe7b449c94281bb12fdadc12be89ee36f447c79a` | 2026-09-12 |
| #59 | `task/generalized-v2-benchmark-evaluation` | `786e8218a00954f7a7f20fe1dfca93ec71a400d4` | `ea8778d5010fcfccec357c57c2861606ecb58bbc` | 2026-09-13 |
| #60 | `task/generalized-v2-benchmark-doc-lock` | `6cbc4a60d3022b51776a4c027bdfc55beedbadac` | `ae42cb01677f94868b2873008d87be677e31f0c8` | 2026-09-13 |

### 3.1 Integration facts recorded only in the former handoff

- PR #32's merge `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96` has tree
  `8a0b7a0aa9e65ebf01fce99c9b27ee25886ba7a6` (committed 2026-08-23 13:32:11 Asia/Jerusalem).
- PR #33's merge `76abdc480e80a84f1503208730d4525cd5e89b69` has tree
  `237325d2c2a41950eab103a8b08c9442e5c9fa97` (committed 2026-08-23 15:34:35 +0300).
- PR #49's reviewed candidate and merge share the tree
  `1b944749fdf52ef3d2175e4437428df4ffc0b656` (merged 2026-09-02 13:26:52 Asia/Jerusalem).
- PR #50's reviewed candidate and merge share the tree
  `88f3ce73c42f0c0680e1d62411816606b2b36dda` (merged 2026-09-02 16:40:45 Asia/Jerusalem).
- The GENERALIZED-V1 Task-5 stack was integrated in the order PR #42 → PR #43 → PR #44, with
  PR #43 and PR #44 retargeted to `main` and exact-base re-reviewed with unchanged heads before
  each merge.

## 4. Closure narratives

- **The three research-validity defects the first short probe exposed — DEFECTS A, B and C
  are ALL CLOSED: implemented, reviewed, approved and merged.** *Historical workflow
  context, kept because it is how these three were run and is not a new prohibition:* the
  SEQUENTIAL-DEFECT POLICY made the DEFAULT breakdown A, then B, then C — each its own
  separately reviewed, separately locked task — with the probe rerun a task of its own,
  never folded into a defect fix; bundling two defects required FOCUSED RECON proving
  them technically INSEPARABLE plus an explicit GPT / user decision, and was never
  authorized. All three closed under that default sequence.
  - **Defect A — `SELF_PRESERVATION_ABORT` was node-scoped, not an ego-global abort:
    CLOSED / MERGED / APPROVED.** Approved `d56fda6`, integrated by `f094e0b` (PR #17) —
    the lock and its evidence are in §7, the contract in §5 Stages 4 and 5. It changed
    abort SEMANTICS only; the `k × 3` action surface, PPO, reward, fuel-damage mechanism
    and BLADE are untouched.
  - **Defect B — PREMATURE ATTACK RE-FIRE EXHAUSTED WEAPONS: CLOSED / APPROVED /
    MERGED.** Approved `39a16f2`, integrated by `60a82d1` (PR #19) — the lock and its
    evidence are in §7, the contract in §5 (Execution, Stage 1) and the routing in §6.
    *The defect, historically:* `GraphPlanExecutor` armed a FIXED 60-tick confirmation wait
    for every salvo (`kill_confirm_ticks` was constructor-configurable but no caller passed
    it), so a slower salvo still in flight could let that wait expire and a redundant second
    salvo consume the last weapons — measured in the first short probe's `post_update`
    damaged eval seed `1000003`, where a B-2 reached its final known target with ZERO
    onboard weapons and then loitered to fuel exhaustion. *The merged correction:* the wait
    is DERIVED per salvo from the ACTUAL auto-selected live weapon and the CURRENT
    engagement distance, with the configured value kept as its FLOOR and FALLBACK — the
    default was NOT merely raised. Lethality, the two-argument attack command and the
    FROZEN vendored BLADE engine are unchanged, and a probabilistic-miss /
    weapons-exhaustion redesign remains OUT of scope.
  - **Defect C — RTB ISSUANCE is not physical RTB COMPLETION: CLOSED / APPROVED /
    MERGED.** Approved `ea62e4e`, integrated by `0de9f21` (PR #21) — the lock and its
    evidence are in §7, the contract in §4 and §5 (Execution, Stage 1, and the tick loop)
    and the routing in §6.
    *The defect, historically:* `GraphPlanExecutor.is_done()` treated the `rtb_issued`
    lifecycle LATCH as RTB-resolved and `run_episode` stopped when it became true, so an
    episode could end while the aircraft was still airborne — measured in the first short
    probe's `post_update` damaged eval seed `1000000`, which recorded `dead=0` and reward 0
    for an ego that could not physically reach home. *The merged correction:*
    `is_done(observation)` requires the LIVE post-step observation; assignment completion
    still comes from executor semantic state, while the PHYSICAL half comes from
    `_physical_state` (airborne / landed / removed) and `_note_dead` reconciles a death on
    the ride home into `executor.dead` before the verdict, so the unchanged reward formula
    receives the truthful terminal loss. `rtb_issued` keeps its ONE job as the single-issue
    toggle guard, and an ego that has committed to return leaves Phase 1 while peers
    continue. The vendored BLADE engine is unchanged.
  **Gating, as it now stands:** the vendored BLADE engine stays FROZEN unless separately
  authorized (§2). The ONE authorized corrected-cell short-probe rerun **HAS been executed
  AND independently reviewed** — `training_output_20260816_162130` at
  `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` — and all three of these defects are
  OPERATIONALLY WITNESSED in its real playback, not only in proof tests. **That witnessing
  survives intact**; what did NOT survive is that run's scientific verdict, which a LATER
  roster/data-integrity review superseded (§7, and the first §8 bullet above). The distinction
  is exact: Defects A, B and C are about what the SIMULATION did, and the roster defect is
  about which targets the MEASUREMENT counted. **A FIRST long baseline was run and was
  scientifically INCONCLUSIVE for that separate reason; the roster defect was then corrected
  and the authorized rerun PASSED the validity gate**, so the cell now has a valid
  measurement (`737b4bf`, §7) and nothing about these three defects is outstanding. Neither
  short probe's numbers, and none of the invalid first long baseline's numbers, are a
  baseline expectation — the approved rerun is the only scientific baseline.
- **A FOURTH, SEPARATE defect — the ROSTER read an ALLOCATION as a WORLD INVENTORY:
  CLOSED / APPROVED / MERGED.** Approved `36365f2`, integrated by `f37ea1c` (PR #24, tree
  `f8015380`) — the lock and its evidence are in §7, the contract in §5 (the Stage-0
  "WORLD INVENTORY IS NOT ORACLE ALLOCATION" block and the roster-integrity block), the
  routing in §6. **It is NOT a regression in Defects A, B or C** — their corrections remain
  merged, witnessed and untouched. *The defect, historically:* `_episode_target_roster`
  answered "which targets does this episode contain?" from `ctx.beliefs` and
  `ctx.oracle_tasks`, both ALLOCATED-ONLY by `solve_and_normalize`'s contract, so any target
  the solver left unselected was missing from the roster while still in the world the
  executor flew through — and the episode was then FAILED for that self-inflicted
  discrepancy AS AN ACCOUNTED `setup` FAILURE. *The merged correction:* the world comes from
  the two RAW pre-solve snapshots `EpisodeContext.known_target_ids` / `executed_target_ids`;
  the beliefs are a SUBSET constraint, not a denominator; `_require_scheduled_cell` checks
  the scheduled cell before anything is paid for; and a roster/world-integrity fault is a
  `MeasurementIntegrityError` that ABORTS the run as INFRASTRUCTURE instead of shrinking a
  scientific denominator. Reward, PPO, the oracle allocation, fuel damage, B2, the seeds,
  the schedules, the tick loop, the executor, the generator and FROZEN BLADE are all
  unchanged. **Consequence, and it is the reason this bullet exists:** the two measurements
  of the merged cell that PRECEDED this correction are permanently and scientifically
  INCONCLUSIVE (§7). The correction is what made a sound measurement possible: the rerun that
  followed it PASSED the validity gate and is the cell's valid baseline (`737b4bf`, §7).
- **`RolloutConfig`/`TrainConfig` construction-default divergence — CLOSED by B1
  (`d6758ac`).** `RolloutConfig` now mirrors `TrainConfig`'s reference-cell fields
  field-for-field (`num_agents`, `n_known`, `n_hidden`, `min_target_distance_km`,
  `min_known_separation_km`, `include_sams`, `randomize_red_airbase_positions`,
  `stretch_target_ratio`) and validates them the same way, as `run_rollout`'s FIRST
  statement. Diagnostic rollouts and training runs now build the same default world.
> **Repository hygiene / documentation alignment: CLOSED** (Grade C, approved candidate
> `52064c2d306df7c8447d159df20e6e189a59bf85`, integrated by
> `5f78904e3af1e2e47386c9b0e01ddbaa273724f5`, PR #12 — see the §7 entry). Verified before
> each removal with exhaustive `git grep`.
> **`README.md` fully replaced** from current repository truth: every path in its layout
> tree exists, every command was run locally before being documented, and the stale-term
> scan (`MAPPO`, `CTDE`, `ActorCriticNetwork`, `30 features`, `centralized critic`,
> `strike_training_2v3`, `plan_editor.py`, `decision-interval`) leaves only the
> §3 invariant stating there is NO centralized critic and one clearly historical sentence.
> **`docs/BLADE_API_DOCUMENTATION.md` audited against the vendored fork** and corrected:
> `blade/__init__.py` exports ONLY the gym registration, so `from blade import Game` binds
> the MODULE and fails later as `'module' object is not callable` — the module-path import
> is now documented; `Scenario.load_from_file` / `Scenario.from_json` DO NOT EXIST and were
> removed in favour of the real `Game(current_scenario=Scenario())` + `game.load_scenario(
> json_string)` path; the gym class is `BLADE` taking `game=`, not `BladeEnv` taking
> `scenario_file=`; `add_strike_mission` is really `create_strike_mission`;
> `Scenario.is_hostile`'s second parameter is NAMED `target_id` but is not a unit id — it
> delegates to `Relationships.is_hostile`, which tests membership in
> `hostiles[side_id]`, and in this fork's scenarios that maps side id → hostile SIDE ids
> (every engine call site passes another unit's `.side_id`); passing a unit id returns
> `False` SILENTLY, so resolve `target.side_id` first; `get_next_coordinates` takes
> origin/destination/speed, not bearing/distance; detection and weapon engagement ranges are
> NAUTICAL MILES while `get_distance_between_two_points` returns KILOMETRES; `DoctrineType`
> has no `ATTACK_HOSTILE`; and the dead `blade_executor_minimal` / `execute_plan`
> integration section was replaced by the real `GraphPlanExecutor` one. The two additive
> fork APIs and the "set `current_scenario.name` BEFORE `start_recording()`" rule are
> documented as contracts. Every claim was machine-checked against the fork.
> **Scenario set reduced to the one active template** (user decision): `close_scenario.json`,
> `far_scenario.json`, `match-aou_demo_2agents.json` and `strike_training_2v3.json` deleted;
> only `data/scenarios/strike_training_4v5.json` remains tracked, and it was NOT modified.
> No code or test referenced any of the four — the sole references were in the old README.
> **Dead utility symbols removed:** `rl/shared_utils.py` loses `nm_to_km` and
> `normalize_value` (zero references across all tracked files; only `haversine_distance` and
> `clip_to_01` are live, both consumed by `graph_builder`), and its module docstring now
> describes only what it provides. `requirements.txt` lost its stale `MAPPO` comment;
> dependency membership and versions are unchanged. No runtime behaviour changed.
> **Flat-path cleanup phase: CLOSED.** All four steps are locked (§7: `814734e`, `d9b8c17`, `ab54ac3`, `7f324fd`), plus a final doc sweep as a coda. The 38 deleted paths are preserved on TWO DISTINCT refs — branch `flat-final` (`4d44c34`) and the annotated tag `pre-cleanup` (commit `561b7cb`; `git rev-parse pre-cleanup` returns the TAG OBJECT `cce4e1e`, so peel it with `pre-cleanup^{commit}`). Nothing in `src/` or `tools/` references the flat path. `LOGS_GUIDE.md`, `RUN_SUMMARY.md`, `docs/MATCH_AOU_API.md`, and `docs/INTEGRATION_GUIDE.md` were **deleted in the final sweep** (superseding the earlier decision to keep the first two as run-log records — the run logs live on the preserved refs, and `train_full` prose in `main` was more confusing than useful). **Both remaining documentation debts are now CLOSED** by the repository-hygiene task that follows `2a3f89c` (§7): `README.md` was REPLACED outright — written from current repository truth, with the MAPPO/CTDE/flat-observation prose gone except one explicitly historical sentence — and `docs/BLADE_API_DOCUMENTATION.md` was AUDITED against the vendored fork and rewritten where it was wrong.
