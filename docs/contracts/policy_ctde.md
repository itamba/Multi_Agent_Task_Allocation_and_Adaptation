# Policy and CTDE contract — observation, encoder, actions, plan effect and the training-only critic

> **Read this when** you change or review the actor's graph observation, the encoder, the
> action head, the legality mask, sampling or re-scoring, how a meta-action edits a plan, the
> Phase-B CTDE critic, its central observation, GAE / value semantics, capture timing,
> checkpoints, or anything that could let privileged or peer information reach the acting path.
>
> **Status: normative, current technical contract** for the code on `main`. Its provenance and
> lock history are in [`implementation.md`](../history/implementation.md) and
> [`documentation_migration.md`](../documentation_migration.md); current run and evidence state
> is in the [handoff](../../graph_rl_project_handoff.md). The no-communication invariants these
> layers serve are [`CLAUDE.md` §3](../../CLAUDE.md#3-architecture--the-load-bearing-invariants).
>
> Related contracts: [runtime](runtime.md) · [reward and solvers](reward_solvers.md) ·
> [training and benchmarks](training_benchmarks.md) · [artifacts and metrics](artifacts_metrics.md).

## 1. Graph observation (Stage 3)

**Build (Stage 3) — `rl/observation/graph_builder.py`.**
`build_graph_observation(scenario, agent_id, current_plan=None, current_time=0, tasks=None, solution=None, precedence_relations=None, config=None) -> GraphObservation`. Stateless projection of `(world, solution)`. `task_features[k, TASK_FEATURE_DIM]` (=6: utility, dist-to-ego, capable, reachable, probability, **sensed**; `TASK_FEATURE_DIM` is the single source of truth the encoder imports), `agent_features[a,1]` (fuel_norm: REAL for ego, `0.0` for peers), COO `edge_index`/`edge_type` over the `EdgeType` IntEnum, `time_norm`. **Relations — builder capability versus current runtime.** The builder CAN construct `PRECEDENCE` edges (task → task, one per `(a, b)` in a non-empty `precedence_relations`), but every current actor-runtime call site in `graph_tick_loop` passes `precedence_relations=[]`, so the actor graphs the current runtime produces carry **`ASSIGNMENT` edges only**. `SPATIAL` is reserved/unused in the actor graph (sensing moved to the `sensed` column). The CTDE central graph's agent → target `SPATIAL` relation is a separate, training-only construct ([§4](#4-phase-b-ctde)). Agent set = `ego ∪ assigned same-side peers`. Requires the ego **airborne** (raises otherwise — always satisfied since build only follows a wake, which requires sensing, which requires airborne).

## 2. Encoder, action head and selection (Stage 4)

**Encode + decide (Stage 4) — `rl/agent/graph_encoder.py` + `rl/action/graph_action.py`.**
`GraphEncoder.forward(obs, edge_attr=None) -> Tensor[k, embed_dim]` — per-task-node embeddings (NOT pooled), single-graph (no batch dim). Defaults `model_dim=64, embed_dim=64, num_heads=4, num_layers=2, task_feat_dim=TASK_FEATURE_DIM`. Edge-masked symmetrized multi-head attention (torch/numpy only, no PyG/DGL) over `forward + reversed + SELF_LOOP` edges with a learned per-relation `type_bias`; learned TASK/EGO/PEER role embedding (node-typing done HERE, reserved MISSION 4th role); injected `time_norm`; self-loops guarantee no empty-softmax NaN. `pool()` = mean over nodes → the size-agnostic **critic hook**, now CONSUMED by the Phase-B `CentralCritic` (its own SECOND `GraphEncoder` instance + `ValueHead`; the ACTOR's encoder and head are unchanged and carry no value head — see the CTDE contract below). `edge_attr` accepted but `None` today (reserved for expected-exec-time on ASSIGNMENT edges). — `ActionHead(embed_dim, hidden_dim=64, num_meta_actions=3).forward([k,embed]) -> [k,3]` emits the per-node SOURCE scores `z[v, m]`. `build_action_mask(obs, ...) -> [k,3]` states per-CELL legality (hard physical/structural: PLAN always; `OPPORTUNISTIC_ENGAGEMENT` iff `unassigned & sensed & capable & reachable`; `SELF_PRESERVATION_ABORT` iff `assigned_to_ego`) and is the SOURCE of semantic-leaf legality, not the action space. **Meta-actions (3):** `PLAN_COMPLIANCE`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT` (Cooperative-Recovery removed — handled upstream by the peer-overdue trigger).

**SELECTION CONTRACT — THE SEMANTIC ACTION REPRESENTATION `semantic_k_plus_2_logmeanexp_v1`
(`graph_action.ACTION_REPRESENTATION_ID`; user-approved 2026-09-16,
[`decisions.md` §1](../history/decisions.md#1-decision-log)).** It supersedes the node-indexed
selection identity locked by Defect A (`d56fda6`), under which every meta-action was selected,
stored and re-scored as one of `k × 3` cells; that representation is historical and is what every
run and checkpoint before this change used.

- **ONE categorical over `k + 2` SEMANTIC LEAVES, in a fixed order:** leaf 0 global
  `PLAN_COMPLIANCE`, leaf 1 global `SELF_PRESERVATION_ABORT`, leaf `2 + i`
  `OPPORTUNISTIC_ENGAGEMENT(task_i)`. It is derived from the EXISTING `k × 3` source scores by
  count-normalized collapse — **no new actor head, no new actor input, encoder unchanged**:
  - `s_PLAN = logsumexp_v z[v, PLAN] − log k` (exact `logmeanexp` over all `k` nodes);
  - `s_ABORT = logsumexp_{v abort-legal} z[v, ABORT] − log n_abort_legal`, over exactly the
    nodes whose ABORT cell is legal; the ONE ABORT leaf is masked when none is;
  - `s_ENGAGE(i) = z[i, ENGAGE]`, legal iff that cell is legal.

  Count normalization is part of the contract: duplicating equal PLAN / ABORT evidence over more
  nodes buys no probability. No canonical node is chosen for PLAN or ABORT. Illegal leaves are
  exactly `−inf`; dtype, device and gradients follow the source scores. **`k == 0` fails loud**
  (`ValueError`): no pooled or invented global score exists.
- **ONE construction site, `graph_action._semantic_dist`.** `sample_action`, `evaluate_action`
  and `summarize_decision` all route through it. `sample_action(logits, mask, deterministic=False)
  -> (meta:int, node_v:Optional[int], log_prob, entropy)`; deterministic selection is
  `torch.argmax` over the SEMANTIC leaves, so an exact tie resolves in leaf order (PLAN, ABORT,
  ENGAGE by ascending task index) — never a source cell and never an aggregate reconstructed
  afterwards.
- **STORED IDENTITY IS SEMANTIC.** `Transition.(meta_action, node_v)` stores `node_v = None` for
  PLAN and ABORT and the task index for ENGAGE — never a placeholder node.
  `evaluate_action(logits, mask, meta, node_v) -> (log_prob, entropy)` re-scores it through the
  same site (grad mode caller-controlled) and FAILS LOUD (`ValueError`) on a global action
  carrying a node, an ENGAGE without an integer node (booleans refused), an out-of-bounds node or
  meta-action, an ENGAGE whose cell is masked at update time, and an ABORT stored while no abort
  cell is legal. With unchanged weights the re-scored log-prob is bitwise the stored one, so the
  epoch-0 PPO ratio is exactly 1 in BOTH updaters.
- **THE ENTROPY BONUS IS THE SEMANTIC-LEAF ENTROPY** (masked-safe clamp form); there is no alias
  spread left to reward.
- **PERMUTATION AND SIZE.** Permuting task nodes leaves the PLAN and ABORT scores and probabilities
  unchanged and permutes the ENGAGE leaves (and a deterministic ENGAGE choice) with their nodes.
  **Total ENGAGE mass still depends on how many genuinely distinct ENGAGE actions exist and their
  scores** — deliberately not changed here ([§6](#6-known-limitations-and-open-items)).
- **SELECTION IDENTITY IS STILL NOT EFFECT SCOPE:** `PLAN_COMPLIANCE` performs NO plan edit;
  `OPPORTUNISTIC_ENGAGEMENT` has a NODE-LOCAL effect; `SELF_PRESERVATION_ABORT` has an EGO-GLOBAL
  effect (Stage 5). `build_action_mask`'s per-cell rules, `NUM_META_ACTIONS` and the `[k, 3]`
  score / mask shape are unchanged.

## 3. Plan effect (Stage 5)

**Effect (Stage 5) — `rl/action/graph_effect.py`.**
`apply_meta_action(solution, obs, ego_id, meta_action, node_v, tasks) -> new_solution`. PURE (BLADE-free; torch is loaded only transitively through the `MetaAction` import), copy-on-write (`_copy_solution`, never mutates input). comply = no-op; engage = add an ego→task assignment AT THE SELECTED NODE. **NULLABLE-NODE GUARDS:** `PLAN_COMPLIANCE` and `SELF_PRESERVATION_ABORT` REQUIRE `node_v is None`; `OPPORTUNISTIC_ENGAGEMENT` REQUIRES an integer node in `[0, len(tasks))`; anything else raises `ValueError`. **ABORT IS EGO-GLOBAL (effect locked by Defect A, `d56fda6`):** the ONE semantic `SELF_PRESERVATION_ABORT` action clears **ALL** of the acting ego's REMAINING assignments. Only `solution[str(ego_id)]` is written: **peer assignment slices, peer beliefs and every task list stay untouched**, `tasks` remains append-only, and `GraphPlanExecutor.done` is not reset. An ego with no key already has an empty mission, so the dict SHAPE is preserved as found (no key is invented). The layer stays PURE and **issues no BLADE command of any kind**: `graph_tick_loop._wake_decision` resyncs ONLY the acting ego's executor slice, and the resulting EMPTY PLAN reaches `GraphPlanExecutor.next_actions` in **Phase 2 of the SAME tick** — the wake, this plan edit and the resync all happen in Phase 1, before any `env.step` — where the PRE-EXISTING empty-plan branch emits the single latched `aircraft_return_to_base`. Nothing new was built for RTB. Does NOT touch the graph — the edge appears on the next rebuild.

## 4. Phase-B CTDE

**PHASE-B CTDE — the TRAINING-ONLY centralized critic —
`rl/observation/central_graph_builder.py` + `rl/training/graph_ppo.py` (its CTDE classes)
+ `rl/training/graph_tick_loop.py` + `rl/training/graph_train.py`.**

Implemented by PR #30. What follows is the implemented contract, derived from the code, not a
design proposal. **No CTDE benefit over actor-only is established by any repository document**,
and none may be pre-claimed from this contract; how CTDE results are reviewed and compared is
[`experiments.md` §4](../workflows/experiments.md#4-run-review--validity-before-performance).

- **TWO TRAINING MODES, SELECTED BY `TrainConfig.training_mode` AND BY NOTHING ELSE.**
  `TRAINING_MODES` = (`actor_only`, `ctde`); `actor_only` is the DEFAULT. The ONE predicate
  behind every branch is `TrainConfig.ctde_enabled`, which reads `training_mode` and
  nothing else. **`value_coeff` IS NOT A MODE SELECTOR**: under `training_mode='ctde'`,
  `validate()` REJECTS `value_coeff <= 0` outright, because a run so configured would build
  central observations and take its advantages from a critic it never trains — neither
  reference algorithm, and recorded as CTDE either way. `validate()` also bounds
  `gae_lambda` to `[0, 1]` and requires `critic_lr > 0`; on an `actor_only` run the unused
  CTDE block may hold any value and is not validated.
- **`actor_only` IS PRESERVED, NOT EMULATED.** It constructs NO critic, NO
  `CentralStateRecorder`, NO `CTDEBuffer`, NO `CTDEUpdater` and NO `CTDEEpisodeRecord`, and
  it computes no central observation, value loss or CTDE advantage. The keyword-omission
  helpers `_ctde_kwargs` / `_central_kwargs` return `{}` rather than a `None`-valued
  keyword, so `_run_one_episode` and `run_episode` are called with EXACTLY their pre-CTDE
  arguments — the stronger invariance claim, and the same pattern `_artifact_kwargs`
  already used. `graph_ppo`'s actor-only half (`EpisodeRecord` / `PPOBuffer` /
  `compute_returns_and_advantages` / `PPOUpdater`) is BYTE-UNCHANGED. This is proven by a
  POISON test: every central-CTDE construction site is replaced by a raiser and an
  `actor_only` run still completes, with a companion CONTROL that flips the mode and shows
  the poison really fires.
- **DECENTRALIZED EXECUTION IS UNCHANGED IN BOTH MODES.** The runtime actor path is still
  `private ego GraphObservation → GraphEncoder → ActionHead → mask → sample`. No central
  state, no peer privileged state, no critic value and no critic parameter reaches action
  selection or `evaluate_action`: `CTDEUpdater._forward_logits` re-encodes the stored
  PRIVATE `tr.gobs` and nothing else, and the advantage crossing from critic to actor is a
  plain python float. Both updaters re-score the SAME semantic identity through the SAME
  `graph_action.evaluate_action` ([§2](#2-encoder-action-head-and-selection-stage-4)). `evaluate` takes NO critic argument and constructs neither a critic
  nor a recorder — held-out evaluation is actor-only in both modes — and a CTDE-trained
  actor runs with the critic object absent. The no-communication invariants
  ([`CLAUDE.md` §3](../../CLAUDE.md#3-architecture--the-load-bearing-invariants)) are not
  weakened by centralized TRAINING.
- **ARCHITECTURE — ACTOR AND CRITIC SHARE NOTHING.** `CentralCritic` owns its OWN
  `GraphEncoder` INSTANCE (the same class, constructed with the CENTRAL feature widths —
  all three were already constructor parameters, so the encoder itself was NOT changed) plus
  its own `ValueHead`, and `CTDEUpdater` builds a SECOND Adam over the critic's parameters
  alone. The two parameter sets are DISJOINT — no sharing, tying or copying — and the actor
  loss and the value loss are backpropagated in TWO SEPARATE `backward()` calls, each with
  its own grad-norm clip and its own `optimizer.step()`. `ValueHead` is a
  `Linear → Tanh → Linear` MLP over the pooled `[embed_dim]` summary, orthogonally
  initialized (hidden at the default `sqrt(2)` gain, OUTPUT at `std=1.0`, the conventional
  value-head gain), so the untrained critic is an arbitrary small-magnitude function of the
  state and **NOT zero everywhere** — nothing relies on it being zero, because a uniform
  offset cancels in the batch-mean subtraction of `compute_ctde_advantages`.
- **THE CENTRAL GRAPH IS THE LIVE WORLD, AND PRESENCE IS LIVENESS.**
  `build_central_graph_observation(scenario, *, agent_ids, executor, current_time, config, acting_agent_id)`
  is STATELESS, like the actor builder, and returns a `CentralGraphObservation` —
  a DISTINCT type, not a `GraphObservation` and not a subclass of one, carrying NO
  `agent_id` field, so a central state can never be mistaken for an actor state.
  - Task nodes are one per LIVE enemy target, enumerated through the SAME
    `generate_all_enemy_tasks` current-world extraction episode setup uses — so the
    inventory is the RAW LIVE WORLD and an unallocated target is present. **`oracle_tasks`
    is NOT read**, which is the roster-integrity contract (Stage 0) honoured rather than
    repeated. A destroyed target simply has no node; there is no dead/alive flag.
  - Agent nodes are one per originally-scheduled same-side agent that is physically alive,
    in the caller's scheduled order. `live_aircraft` collapses the executor's own three-way
    classification: airborne (in `scenario.aircraft`) or landed (in some `airbase.aircraft`
    inventory) is LIVE; absent from both is dead and loses its node. **RTB ISSUANCE AND
    LANDING ARE NOT DEATH** — an ego ordered home keeps its node even though Phase 1 stops
    processing it.
  - **THE CENTRAL STATE IS DECISION-CONDITIONED, BY ROLE ONLY.** Every actual decision
    capture names the agent that OWNS the current decision (`acting_agent_id`, the tick
    loop's waking `ego_id`); the builder locates it by IDENTITY among the live agent nodes
    and sets `ego_index` to its GLOBAL node index `k + row`, so the shared encoder's
    EXISTING role mechanism marks that node EGO and every other live agent PEER. The critic
    therefore values `V(global_state, acting_ego)` rather than `V(global_state)`. **This is
    the only conditioning:** the physical features, edges, node set and feature widths are
    identical whichever agent acts; no numeric or string agent identity, agent order,
    severity, condition label, wake kind, selected action, reward or future information is
    added; and the encoder, `pool()` (mean pooling), `CentralCritic` / `ValueHead`, the
    critic optimizer, PPO, GAE and the reward are unchanged. The conditioning follows the
    physical agent, not a fixed row or the scheduled order. **It FAILS CLOSED:** a capture
    whose acting agent has no live node (never scheduled, or physically dead) raises and
    records nothing — there is no silent fallback. `CentralStateRecorder.capture` requires
    `acting_agent_id`; only a NON-decision projection (`build_central_graph_observation`
    with `acting_agent_id=None`) carries the `NO_EGO_INDEX` (`-1`) sentinel, under which
    every agent node keeps the same role. It is training-only privileged conditioning: the
    actor's observation, mask and selection are untouched and never see the central state.
    Every CTDE run measured before this conditioning was introduced — including both
    actor-gradient diagnostics at `6ed964a1abd09de2130aee3d0d314c8f32165056` — used the
    earlier SYMMETRIC central state (`ego_index` always `NO_EGO_INDEX`, no distinguished
    agent) and remains a description of that critic.
  - **FEATURES, exactly as implemented.** `task_features[k, 2]` = `[utility_norm,
    probability]` (`CENTRAL_TASK_FEATURE_DIM`). `agent_features[a, 1]` = `[fuel_norm]`
    (`CENTRAL_AGENT_FEATURE_DIM`) — **REAL for EVERY live agent**, which is the exact
    asymmetry the actor graph must not have (peer fuel is unsensable under
    no-communication; the point of a centralized critic is that TRAINING may read it).
    `time_norm` is the actor's own `current_time / max_sim_ticks`, clipped, from the SAME
    `GraphObservationConfig` the actor builder is using on that episode.
  - **EDGES: the COMPLETE live-agent → live-target bipartite relation**, one
    `EdgeType.SPATIAL` edge each (`CENTRAL_EDGE_TYPE`; SPATIAL is RESERVED / unused in the
    actor graph, so borrowing the code changes nothing the actor builds), with
    `edge_attr[E, 5]` = `[distance_norm, capable, reachable, sensed, assigned]`
    (`CENTRAL_EDGE_ATTR_DIM`). `reachable` IMPORTS the actor's own
    `graph_builder._reachable_by_ego` round-trip model rather than reimplementing one;
    `sensed` is privileged ALL-AGENT sensing at the ONE unified `DETECTION_KM`; and
    **`assigned` is CURRENT executor plan membership** — `plan_target_ids` resolves
    `executor.plans[agent]` against `executor.tasks[agent]` with `_resolve_step`'s own
    bounds semantics (a documented MIRROR kept out of that module's import closure, its
    equivalence TEST-ENFORCED against a real `GraphPlanExecutor`), **never from
    `oracle_solution`, never from a private belief, and never from t=0 `A_init` after
    runtime adaptation**. It is plan MEMBERSHIP, not eligibility: no `done` filter and no
    level gating.
  - **PRIVILEGED MEANS "ALL AGENTS, RIGHT NOW" — IT DOES NOT MEAN "THE ANSWER".** The
    critic is deliberately NOT given `oracle_solution` / `oracle_tasks` / `U_oracle` / any
    reward component, the episode seed, the scheduled fuel-damage severity or condition
    label, the known-vs-hidden split, future RNG, or any future outcome. **Do not add a
    feature this list does not name.**
  - **SIZE IS VARIABLE, WITH ONE FLOOR.** There is no padding and no fixed cardinality:
    the encoder is size-agnostic and its self-loops keep an empty edge set safe. `k` (live
    targets) MAY legitimately be **0** — every target destroyed is a normal late-episode
    state. The live-agent count is likewise variable, but **at an ACTUAL DECISION CAPTURE it
    is at least 1**: a decision requires an airborne ego, so that ego always has a node.
    `CentralCritic.forward` does carry an all-empty `n_nodes == 0` guard returning a finite
    zero, but that branch is DEFENSIVE — it makes the output finite by construction rather
    than by an argument about the caller, and it is **not a reachable normal decision
    state**.
- **MULTI-AGENT TEMPORAL SEMANTICS — ONE CENTRAL STATE PER ACTUAL DECISION.**
  `run_episode(..., central=CentralStateRecorder())` calls `capture` INSIDE the `if wake`
  branch and IMMEDIATELY BEFORE `_wake_decision`, and nowhere else, naming the loop's
  waking `ego_id` as `acting_agent_id` — so sample `i` is the global state the team was in
  when decision `i` was made, BEFORE that decision changed anything, with decision `i`'s
  owner in the EGO role, and `recorder.samples` is aligned 1:1 and index-for-index with
  `EpisodeResult.trajectory`. `CTDEEpisodeRecord` VALIDATES that alignment on construction,
  so a drifted capture seam fails LOUD rather than mispairing a value with a decision.
  **WAKE ORDERING IS STILL SEQUENTIAL, NOT A JOINT SAME-TICK ACTION.** With two egos waking
  on one tick the order is `capture(A) → act(A)+resync(A) → capture(B) → act(B)+resync(B)
  → env.step`: no `env.step` between them, so B's PHYSICAL world equals A's, while B's
  central `assigned` feature legitimately reflects A's already-applied resync. That is
  CAUSAL, not a leak — the critic is centralized by design, and B still DECIDES from its
  own private observation alone. The `central` parameter defaults to `None`, which leaves
  the loop byte-identical to its pre-CTDE behaviour.
- **CREDIT PATH: GAE OVER THE GLOBAL DECISION SEQUENCE.** `compute_ctde_advantages` is the
  CTDE REPLACEMENT for `compute_returns_and_advantages`; it does not call it, and it never
  runs on an `actor_only` run. `V_old` is evaluated ONCE for every sample under
  `torch.no_grad` BEFORE epoch 0 and stays fixed for the whole update, so the regression
  target cannot chase the network fitting it. `compute_gae` runs PER EPISODE over the
  episode's SINGLE ordered decision sequence — **deliberately NOT regrouped per ego**, which
  is what `EpisodeRecord` does for the Phase-A per-ego credit structure — with
  `delta_t = r_t + gamma*V_next[t] - V_old[t]`, `A_t = delta_t + gamma*gae_lambda*A_{t+1}`,
  `target_t = A_t + V_old[t]`, and **`V_next` of the LAST decision is ZERO** (the episode
  genuinely ends there). The ONE GAE pass (`_gae_pass`) also keeps each `delta_t`, stored on
  `CTDEAdvantageBatch.td_residuals` beside the realized `rewards` it consumed; `compute_gae`
  still returns only `(advantages, value_targets)`. Per-decision rewards are READ off the transitions
  (`episode_rewards_sequence`), so the credit math consumes exactly what the unchanged
  terminal reward layer produced. Advantages are normalized across ALL decision samples of
  the batch under the same `adv_norm_eps` guard the actor-only path uses, and the actor
  consumes them DETACHED. The critic takes an MSE value loss scaled by `value_coeff`; there
  is no value clipping in v1. `gamma` comes from `PPOConfig` — deliberately NOT duplicated
  on `CTDEConfig`, so a run has ONE discount factor. **Current defaults, from the code:**
  `critic_lr = 3e-4`, `value_coeff = 0.5`, `gae_lambda = 0.95`.
- **ZERO-WAKE EPISODES.** A zero-wake episode contributes NO actor sample, NO critic sample
  and NO baseline mass to a CTDE update. It remains a **valid scientific episode outcome**,
  never a failure, and keeps its existing reward-diagnostic accounting — a batch with zero
  decisions is the same clean no-op `PPOUpdater` documents, reported with
  `n_epochs_run == 0`.
- **`baseline` KEEPS ITS ACTOR-ONLY MEANING IN BOTH MODES, AND THIS IS LOAD-BEARING.**
  `CTDEUpdater.update` reports `baseline` as the batch's mean EPISODE REWARD (zero-wake
  episodes included) — NOT the critic's mean value, even though the CTDE baseline really is
  the critic. `graph_train` records that key as an iteration's `train_reward_mean`, so
  putting a value estimate there would make one recorded field mean a reward under
  `actor_only` and a value under `ctde`, and the two modes' learning curves would stop being
  comparable while still looking as though they were. The critic's own estimate is reported
  SEPARATELY. A CTDE training record additionally persists the CRITIC's four diagnostics —
  `value_loss`, `value_mean`, `value_target_mean`, `critic_grad_norm` — copied straight out
  of the dict `CTDEUpdater.update` returned and NEVER recomputed; they are added ONLY when
  `ctde_enabled`, so an `actor_only` record is byte-unchanged with those keys ABSENT rather
  than null (a nullable key would invite reading "no critic" as "a critic that scored 0").
  `run_config.json` carries a `training` block: `mode`, `ctde_enabled`, and the resolved
  `ctde` config or `null`.
- **OBSERVATIONAL CREDIT REPORT.** `PPOUpdater.update` and `CTDEUpdater.update` take an
  optional `credit_sink`; after a PRODUCTIVE update (`n_epochs_run > 0`) each hands it ONE
  `CreditReport` carrying the SAME batch object the update consumed (`AdvantageBatch` /
  `CTDEAdvantageBatch`, with identity-only `record_positions` and chain / decision ordinals).
  No forward, GAE pass, RNG draw or gradient is added and nothing reads the report back; the
  trainer turns it into `train_credit_diagnostics.jsonl`
  ([artifacts and metrics §5.1](artifacts_metrics.md#51-training-credit-diagnostics)).
- **OBSERVATIONAL EPOCH-0 ACTOR-GRADIENT REPORT (CTDE only, opt-in).** `CTDEUpdater.update` also
  takes `gradient_group_ids` (one opaque integer per transition, in batch order) together with
  `gradient_sink`. At epoch 0, after the per-transition policy losses and `actor_loss` are built
  and before the real `actor_loss.backward()`, it differentiates each id's summed policy loss
  divided by the full batch size, the total surrogate and the total actor loss with
  `torch.autograd.grad(…, retain_graph=True)`, and after a productive update hands the sink ONE
  `ActorGradientReport` of flat numpy gradients. With the optional `gradient_contrast_ids`
  `(positive, negative)` pair it also forms, when both ids occur, the differentiable contrast
  `mean P(ABORT | positive) - mean P(ABORT | negative)` from the SAME epoch-0 logits through
  `_semantic_dist`, and reports its value and gradient. No forward, GAE pass, RNG draw, `.grad` write or
  optimizer change is added, and the real backward, both clips and both steps are unchanged;
  the updater attaches no meaning to the ids. `PPOUpdater` is not instrumented
  ([artifacts and metrics §5.2](artifacts_metrics.md#52-ctde-actor-gradient-diagnostics)).
- **CHECKPOINTS.** `save_checkpoint(policy, updater, iteration, ckpt_dir, critic=None)`.
  **THE ACTOR-ONLY PAYLOAD** — with `critic is None` (every `actor_only` run) — holds the five
  historical keys (`iteration` / `encoder` / `head` / `optimizer` / `ppo_config`) PLUS
  `action_representation_id`. A CTDE run saves strictly MORE: those six keys (`encoder` /
  `head` / `optimizer` are the ACTOR's) plus `training_mode`, `critic_encoder`, `value_head`,
  `critic_optimizer` and `ctde_config`. **SEMANTIC COMPATIBILITY IS INTENTIONALLY BROKEN:** the
  encoder / head tensor shapes did not change, so a historical checkpoint (five keys, no
  representation id) would still load into them, but its weights were trained under the retired
  node-indexed action representation and it remains evidence of that representation only. **No
  migration, warm-start conversion, loader compatibility or resume exists**, in either mode;
  restoring a run remains a separate deferred task. There is deliberately NO second "actor
  export" file.
- **PRESETS.** A preset may set `training_mode` and a nested `"ctde"` block (the sibling of
  `"ppo"`), read only by a `ctde` run. The CTDE block has NO CLI flags of its own — it is
  deliberately a preset-only layer, so there is no second naming scheme to drift from
  `CTDEConfig`. **No CTDE preset exists in the repository**; a run that needs one defines it
  under its authorized plan.
- **SCIENTIFIC NON-CLAIMS, BINDING.** The proof tests, the module `_selftest`s and a passing
  suite are ENGINEERING evidence and measure nothing scientific. **No CTDE benefit — in
  reward, survival, sample efficiency, behavioural separation or anything else — is
  established by this contract or may be pre-claimed.** A CTDE claim needs a reviewed
  comparison under the design's own review
  ([`experiments.md` §4](../workflows/experiments.md#4-run-review--validity-before-performance)).

### 4.1 Status of the CTDE layer

- **`actor_only` REMAINS THE DEFAULT AND THE PRESERVED REFERENCE PATH.** A run that does
  not select `ctde` constructs no critic, no central observation, no value loss and no
  CTDE advantage — the Phase-A path is not emulated, it is simply the one that runs.
  Preserving it is load-bearing: the fixed-cell baselines in
  [`measurements.md`](../history/measurements.md#1-run-registry) were measured on it.
- **CTDE MEASUREMENTS AND THEIR SCOPE.** An old fixed-cell CTDE measurement exists and is out of
  scope unless the user asks ([`experiments.md` §4.4](../workflows/experiments.md#44-comparator-discipline)).
  A GENERALIZED-V2 CTDE development-profile run and its review status are listed in the
  [handoff](../../graph_rl_project_handoff.md#4-runs-and-evidence--current-references). Neither
  establishes a CTDE benefit in this contract.

## 5. Code routing

Research-validity changes (the critic's inputs, the actor/critic boundary, capture timing,
actor-only preservation) follow [`cc_review.md` §4](../workflows/cc_review.md#4-risk-and-verification).

| Task | Files and symbols | Contract |
|---|---|---|
| select a training mode (configuration, not a contract change) | `rl/training/graph_train.py`: `TrainConfig.training_mode`, `TRAINING_MODES`, `TrainConfig.ctde_enabled`, the `"ctde"` preset block over `CTDEConfig` | §4 |
| change what the critic sees | `rl/observation/central_graph_builder.py`: `CentralGraphObservation`, `build_central_graph_observation`, `CentralStateRecorder`, `live_aircraft`, `plan_target_ids`, `NO_EGO_INDEX`, `CENTRAL_TASK_FEATURE_DIM`, `CENTRAL_AGENT_FEATURE_DIM`, `CENTRAL_EDGE_ATTR_DIM`, `CENTRAL_EDGE_TYPE` (pure: no torch, BLADE or gym import; never imports `graph_episode_setup`) | §4, exclusion list |
| change the actor/critic boundary or GAE / value semantics | `rl/training/graph_ppo.py`: `CTDEConfig`, `ValueHead`, `CentralCritic`, `build_central_critic`, `CTDEEpisodeRecord`, `CTDEBuffer`, `compute_gae`, `_gae_pass`, `compute_ctde_advantages`, `CTDEUpdater`, `episode_rewards_sequence`; tests `tests/test_graph_ctde.py`, `tests/test_graph_ppo.py` | §4 |
| change what an update reports about its credit | `rl/training/graph_ppo.py`: `CreditReport`, `CreditSink`, the `credit_sink` parameter of `PPOUpdater.update` / `CTDEUpdater.update`, `AdvantageBatch.record_positions` / `chain_ordinals`, `CTDEAdvantageBatch.rewards` / `td_residuals` / `record_positions` / `decision_ordinals`; tests `tests/test_graph_semantic_action_credit.py` | §4; [artifacts and metrics §5.1](artifacts_metrics.md#51-training-credit-diagnostics) |
| change the epoch-0 actor-gradient report | `rl/training/graph_ppo.py`: `ActorGradientReport`, `GradientSink`, `_flat_actor_grad`, the `gradient_group_ids` / `gradient_sink` / `gradient_contrast_ids` parameters of `CTDEUpdater.update`; tests `tests/test_graph_ctde_actor_gradient_diagnostics.py` | §4; [artifacts and metrics §5.2](artifacts_metrics.md#52-ctde-actor-gradient-diagnostics) |
| change when the central state is captured | `rl/training/graph_tick_loop.py`: `run_episode(central=...)` and its `capture` call immediately before `_wake_decision` | §4; [runtime §5](runtime.md#5-resync-stage-6-and-the-two-phase-tick-loop) |
| change actor-only preservation or checkpoints | `rl/training/graph_train.py`: `_ctde_kwargs`, `_central_kwargs`, `save_checkpoint(..., critic=None)`, the critic diagnostics on training records, `run_config.json:/training`; poison test and control in `tests/test_graph_ctde.py` | §4 |
| change the graph representation | `rl/observation/graph_builder.py`: `GraphObservation`, `GraphObservationConfig`, `EdgeType`, `TASK_FEATURE_DIM` | §1 |
| change the encoder (one class, instantiated by the actor and the critic) | `rl/agent/graph_encoder.py`: `GraphEncoder`, `pool()` | §2, §4 |
| change actions, mask, sampling or re-scoring | `rl/action/graph_action.py`: `MetaAction`, `ACTION_REPRESENTATION_ID`, `ActionHead`, `build_action_mask`, `_semantic_dist`, `semantic_leaf_index`, `semantic_leaf_identity`, `GLOBAL_META_ACTIONS`, `sample_action`, `evaluate_action`; `rl/training/graph_tick_loop.py`: `Transition.node_v`; tests `tests/test_graph_action_evaluate.py`, `tests/test_graph_semantic_action_credit.py` | §2 |
| change how a decision edits the plan | `rl/action/graph_effect.py`: `apply_meta_action` (nullable-node guards) | §3 |

## 6. Known limitations and open items

- **`reachable_by_ego` marginal-detour model:** `graph_builder._reachable_by_ego` is a conservative round-trip placeholder; intended model is marginal detour-cost vs remaining fuel slack (isolated to the builder; the mask reads the column).
- **Total ENGAGE mass under the semantic representation.** PLAN and ABORT are one leaf each, but the combined probability of OPPORTUNISTIC_ENGAGEMENT still depends on the number and scores of the genuinely distinct `ENGAGE(task_i)` leaves. This is not addressed; a hierarchical meta-action / target factorization is a possible future research intervention and is not authorized by this contract.
- **`k == 0` acting is unsupported:** the semantic construction fails loud rather than inventing a pooled global score; a decision requires at least one task node.
