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
`GraphEncoder.forward(obs, edge_attr=None) -> Tensor[k, embed_dim]` — per-task-node embeddings (NOT pooled), single-graph (no batch dim). Defaults `model_dim=64, embed_dim=64, num_heads=4, num_layers=2, task_feat_dim=TASK_FEATURE_DIM`. Edge-masked symmetrized multi-head attention (torch/numpy only, no PyG/DGL) over `forward + reversed + SELF_LOOP` edges with a learned per-relation `type_bias`; learned TASK/EGO/PEER role embedding (node-typing done HERE, reserved MISSION 4th role); injected `time_norm`; self-loops guarantee no empty-softmax NaN. `pool()` = mean over nodes → the size-agnostic **critic hook**, now CONSUMED by the Phase-B `CentralCritic` (its own SECOND `GraphEncoder` instance + `ValueHead`; the ACTOR's encoder and head are unchanged and carry no value head — see the CTDE contract below). `edge_attr` accepted but `None` today (reserved for expected-exec-time on ASSIGNMENT edges). — `ActionHead(embed_dim, hidden_dim=64, num_meta_actions=3).forward([k,embed]) -> [k,3]`. `build_action_mask(obs, ...) -> [k,3]` (hard physical/structural legality; `OPPORTUNISTIC_ENGAGEMENT` gated by `unassigned` AND `sensed`). `sample_action(logits, mask, deterministic=False) -> (meta:int, node_v:int, log_prob, entropy)`. `evaluate_action(logits, mask, meta, node_v) -> (log_prob, entropy)` re-scores a stored decision through the SAME private `_masked_dist` construction site (grad-mode caller-controlled; masked / out-of-bounds cells fail loud). **Meta-actions (3):** `PLAN_COMPLIANCE`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT` (Cooperative-Recovery removed — handled upstream by the peer-overdue trigger).
**SELECTION CONTRACT (locked by Defect A, `d56fda6`).** The action surface REMAINS `k × 3`, and EVERY meta-action retains NODE-INDEXED SELECTION IDENTITY: the selected `(node_v, meta_action)` cell is what `sample_action` samples, what `Transition` stores, and what `evaluate_action` re-scores under PPO, with `node_v` still bounds-checked to `[0, k)`. **Selection identity is NOT effect scope**, and the three members differ on the second: `PLAN_COMPLIANCE` performs NO plan edit; `OPPORTUNISTIC_ENGAGEMENT` has a NODE-LOCAL effect (it assigns the ego to THAT task node); `SELF_PRESERVATION_ABORT` has an EGO-GLOBAL effect (Stage 5). `build_action_mask` governs SELECTION only — its per-column legality rules, `NUM_META_ACTIONS`, the logit/mask shape and the sampling/evaluation action identities are all UNCHANGED by Defect A.

## 3. Plan effect (Stage 5)

**Effect (Stage 5) — `rl/action/graph_effect.py`.**
`apply_meta_action(solution, obs, ego_id, meta_action, node_v, tasks) -> new_solution`. PURE (BLADE-free, torch-free), copy-on-write (`_copy_solution`, never mutates input). comply = no-op; engage = add an ego→task assignment AT THE SELECTED NODE. **ABORT IS EGO-GLOBAL (locked by Defect A, `d56fda6`):** selecting `SELF_PRESERVATION_ABORT` on ANY legal cell clears **ALL** of the acting ego's REMAINING assignments, and **the selected node does NOT scope the effect** — every legal abort cell of a given ego therefore produces the identical empty slice. Only `solution[str(ego_id)]` is written: **peer assignment slices, peer beliefs and every task list stay untouched**, `tasks` remains append-only, and `GraphPlanExecutor.done` is not reset. An ego with no key already has an empty mission, so the dict SHAPE is preserved as found (no key is invented). The layer stays PURE and **issues no BLADE command of any kind**: `graph_tick_loop._wake_decision` resyncs ONLY the acting ego's executor slice, and the resulting EMPTY PLAN reaches `GraphPlanExecutor.next_actions` in **Phase 2 of the SAME tick** — the wake, this plan edit and the resync all happen in Phase 1, before any `env.step` — where the PRE-EXISTING empty-plan branch emits the single latched `aircraft_return_to_base`. Nothing new was built for RTB. Does NOT touch the graph — the edge appears on the next rebuild.

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
  plain python float. `evaluate` takes NO critic argument and constructs neither a critic
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
  `build_central_graph_observation(scenario, *, agent_ids, executor, current_time, config)`
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
  - **THERE IS NO DISTINGUISHED EGO.** `ego_index` is `NO_EGO_INDEX` (`-1`), and the shared
    encoder marks a node EGO only for `0 <= ego_index < N`, so every agent node keeps the
    same role and the graph is SYMMETRIC over live agents. No encoder change was needed and
    none was made.
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
  branch and IMMEDIATELY BEFORE `_wake_decision`, and nowhere else — so sample `i` is the
  global state the team was in when decision `i` was made, BEFORE that decision changed
  anything, and `recorder.samples` is aligned 1:1 and index-for-index with
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
  genuinely ends there). Per-decision rewards are READ off the transitions
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
- **CHECKPOINTS.** `save_checkpoint(policy, updater, iteration, ckpt_dir, critic=None)`.
  **THE ACTOR-ONLY PAYLOAD IS UNCHANGED** — with `critic is None` (every `actor_only` run)
  it holds EXACTLY the five keys it always held (`iteration` / `encoder` / `head` /
  `optimizer` / `ppo_config`), nothing renamed and nothing added, not even a mode label, so
  a Phase-A checkpoint stays readable by anything that could read one. A CTDE run saves
  strictly MORE: the same five keys (`encoder` / `head` / `optimizer` are the ACTOR's) plus
  `training_mode`, `critic_encoder`, `value_head`, `critic_optimizer` and `ctde_config`.
  There is deliberately NO second "actor export" file — the actor portion of the one payload
  already suffices for later inference, precisely because the actor's keys did not move.
  **There is NO loader and NO resume**, in either mode; restoring a run remains a separate
  deferred task, and no export functionality beyond the above exists.
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
| change the actor/critic boundary or GAE / value semantics | `rl/training/graph_ppo.py`: `CTDEConfig`, `ValueHead`, `CentralCritic`, `build_central_critic`, `CTDEEpisodeRecord`, `CTDEBuffer`, `compute_gae`, `compute_ctde_advantages`, `CTDEUpdater`, `episode_rewards_sequence`; tests `tests/test_graph_ctde.py`, `tests/test_graph_ppo.py` | §4 |
| change when the central state is captured | `rl/training/graph_tick_loop.py`: `run_episode(central=...)` and its `capture` call immediately before `_wake_decision` | §4; [runtime §5](runtime.md#5-resync-stage-6-and-the-two-phase-tick-loop) |
| change actor-only preservation or checkpoints | `rl/training/graph_train.py`: `_ctde_kwargs`, `_central_kwargs`, `save_checkpoint(..., critic=None)`, the critic diagnostics on training records, `run_config.json:/training`; poison test and control in `tests/test_graph_ctde.py` | §4 |
| change the graph representation | `rl/observation/graph_builder.py`: `GraphObservation`, `GraphObservationConfig`, `EdgeType`, `TASK_FEATURE_DIM` | §1 |
| change the encoder (one class, instantiated by the actor and the critic) | `rl/agent/graph_encoder.py`: `GraphEncoder`, `pool()` | §2, §4 |
| change actions, mask, sampling or re-scoring | `rl/action/graph_action.py`: `MetaAction`, `ActionHead`, `build_action_mask`, `sample_action`, `evaluate_action`, `_masked_dist` | §2 |
| change how a decision edits the plan | `rl/action/graph_effect.py`: `apply_meta_action` | §3 |

## 6. Known limitations and open items

- **`reachable_by_ego` marginal-detour model:** `graph_builder._reachable_by_ego` is a conservative round-trip placeholder; intended model is marginal detour-cost vs remaining fuel slack (isolated to the builder; the mask reads the column).
