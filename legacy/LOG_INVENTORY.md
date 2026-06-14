# LOG_INVENTORY.md

> Inventory of every logging output (stdout/stderr) and every disk write produced
> by the active project code (`train_full.py` + `src/match_aou/**`).
> Excludes the vendored BLADE engine fork at
> `src/match_aou/integrations/panopticon-main/` (frozen per CLAUDE.md §2) and
> archived DQN-era code under `legacy/` (also frozen). One BLADE print() that
> fires through every recording export is included as the last row of Section 1
> because the researcher will see it in their terminal.

## Deliverables produced by this task

- [LOG_INVENTORY.md](LOG_INVENTORY.md) — this file.
- [run_capture.log](run_capture.log) — full stdout+stderr (13,443 lines) from a
  5-episode runtime capture; kept in repo root for re-reading.
- Files created in `training_output/` during the run — see Section 4.

## Runtime capture: how it was produced

No source files were modified. The capture was produced by passing CLI flags
that proportionally reduce all "every-N" thresholds to 1, so every cadence
path fires every episode for maximum coverage:

```
PATH="C:\Users\Itama\anaconda3\envs\nlp_env\Library\bin;$PATH" \
  python train_full.py \
    --episodes 5 \
    --save-freq 1 \
    --validate-every 1 \
    --record-every 1 \
    --progress-every 1 \
    --verbose \
    --fuel-damage \
    --debug-force-flags timeout,l2-fallback
```

(`PATH` was prefixed with `nlp_env/Library/bin` so the `bonmin` MINLP solver
was discoverable; `python` invoked was the `nlp_env` interpreter per
CLAUDE.md §1.)

Because the change was CLI-only, there is nothing to revert.

---

## 1. stdout / stderr inventory

> Format note: the project routes everything through Python's `logging` module
> (root logger, formatter `%(asctime)s | %(levelname)-7s | %(name)s |
> %(message)s`). Three handlers are attached (train_full.py:2459-2475 +
> 2657-2662): a console `StreamHandler` (level INFO, or DEBUG when
> `--verbose`), a master `FileHandler` mirror at the same level
> (`training_output/logs/training.log`), and a per-episode DEBUG
> `FileHandler` (`training_output/logs/episode_NNNN.log`) attached for the
> duration of each episode. Most of the rows below therefore appear on
> stdout *only* when `--verbose` is set (they always appear in the
> per-episode log file regardless).
>
> Third-party stdout/stderr also flows through these handlers (Pyomo's
> `pyomo.core` and `pyomo.opt` DEBUG logs — visible only with `--verbose`)
> or directly (gymnasium's `passive_env_checker.py:158` UserWarning, which
> goes to stderr). These are not project call sites and are not enumerated
> below, but they appear in the runtime sample in Section 3.

| Source (file:line) | Trigger | Category | Example output (literal or templated) | Notes |
|---|---|---|---|---|
| [train_full.py:181](train_full.py:181) | After loading BLADE env spec | run-init | `BLADE registered max_episode_steps: <N>` | INFO |
| [train_full.py:194](train_full.py:194) | When scenario duration > `--max-ticks` | run-init | `Scenario duration (<D>) > max_steps (<M>). Consider increasing --max-ticks.` | WARNING |
| [train_full.py:199](train_full.py:199) | After `env.reset()` in `setup_blade_env` | run-init | `BLADE env ready: duration=<D>, max_episode_steps=<M>, start_time=<T>, current_time=<T>` | INFO |
| [train_full.py:221](train_full.py:221) | Each `reload_scenario` call (every episode + every flagged-replay) | episode-init | `Reloaded scenario from <path>` | DEBUG |
| [train_full.py:248](train_full.py:248) | `solve_match_aou` called with empty agents/tasks | error | `No tasks or agents to solve` | WARNING |
| [train_full.py:255](train_full.py:255) | MINLP solver returns empty solution | error | `MATCH-AOU returned empty solution` | WARNING |
| [train_full.py:266](train_full.py:266) | After every successful `solve_match_aou` | episode-init | `  → <N> assignments, <K> unselected` | DEBUG |
| [train_full.py:345](train_full.py:345) | `split_tasks` fallback when no observation given | episode-init | `Task split: <P> partial, <F> full, <H> hidden` | DEBUG |
| [train_full.py:362](train_full.py:362) | `split_tasks` when fleet has no radar range | episode-init | `Task split: ... (chain check skipped: no radar range)` | DEBUG |
| [train_full.py:392](train_full.py:392) | `split_tasks` when isolated count exceeds partial budget | error | `Discovery chain (split): isolated=<I> exceeds partial budget=<P>; <K> isolated target(s) will be hidden and undiscoverable` | WARNING |
| [train_full.py:397](train_full.py:397) | After the warning above | episode-init | `Task split: <P> partial, <F> full, <H> hidden` | DEBUG |
| [train_full.py:424](train_full.py:424) | `split_tasks` valid draw (clean or resampled) | episode-init | `Discovery chain (split): clean (hidden=<H>, known=<K>, isolated_pinned=<I>, min_radar=<R> km)` | DEBUG |
| [train_full.py:429](train_full.py:429) | After line 424 | episode-init | `Task split: <P> partial, <F> full, <H> hidden` | DEBUG |
| [train_full.py:445](train_full.py:445) | `split_tasks` exhausted retries | error | `Discovery chain (split): no valid split after <N> attempts; some hidden targets may have no known radar neighbour (min_radar=<R> km)` | WARNING |
| [train_full.py:450](train_full.py:450) | After line 445 | episode-init | `Task split: <P> partial, <F> full, <H> hidden` | DEBUG |
| [train_full.py:629](train_full.py:629) | Every BLADE `handle_aircraft_attack(...)` action issued | mid-episode | `  Tick <T> [EXEC/RL/VAL] ATTACK: agent <ID8>.. → target <ID8>..` | DEBUG; in `_log_blade_action` |
| [train_full.py:638](train_full.py:638) | Every BLADE `move_aircraft(...)` action | mid-episode | `  Tick <T> [<src>] MOVE:   agent <ID8>.. → (<coords>)` | DEBUG |
| [train_full.py:646](train_full.py:646) | Every BLADE `launch_aircraft_from_airbase(...)` action | mid-episode | `  Tick <T> [<src>] LAUNCH: from airbase <ID8>..` | DEBUG |
| [train_full.py:651](train_full.py:651) | Every BLADE `return_to_base(...)` action | mid-episode | `  Tick <T> [<src>] RTB:    agent <ID8>..` | DEBUG |
| [train_full.py:655](train_full.py:655) | BLADE action that doesn't match any of the four regexes | mid-episode | `  Tick <T> [<src>] ACTION: <action[:80]>` | DEBUG |
| [train_full.py:669](train_full.py:669) | `_log_progress` — every `PROGRESS_LOG_INTERVAL`=1000 ticks | mid-episode | `  ── Tick <T> ── airborne: <A>/<N> \| RL decisions: <D> \| reward: <R> \| targets attacked: <K>/<T>` | DEBUG |
| [train_full.py:705](train_full.py:705) | Start of `run_validation_episode` | validation | `--- Validation run (oracle only, no RL) ---` | DEBUG |
| [train_full.py:722](train_full.py:722) | Validation: no agents in scenario | error | `Validation: no agents found, skipping` | WARNING |
| [train_full.py:727](train_full.py:727) | Validation: no tasks in scenario | error | `Validation: no tasks found, skipping` | WARNING |
| [train_full.py:730](train_full.py:730) | Validation: just before solving | validation | `Validation: <N> agents, <T> tasks` | DEBUG |
| [train_full.py:759](train_full.py:759) | Validation solver returned empty | error | `Validation: solver returned empty solution, skipping` | WARNING |
| [train_full.py:786](train_full.py:786) | Per attacking agent, in validation, before launch | validation | `  VAL plan: agent=<ID8> → tasks=[<t1>,<t2>,..]` | DEBUG |
| [train_full.py:806](train_full.py:806) | For each aircraft launched in validation | validation | `  Validation LAUNCH: <name> (id=<ID8>..) from airbase <ID8>..` | DEBUG |
| [train_full.py:863](train_full.py:863) | Validation: each tick an agent goes from airborne→landed | validation | `  Tick <T> VAL RTB: agent <ID8>.. landed` | DEBUG |
| [train_full.py:873](train_full.py:873) | Validation: every 10th tick in last 100 before max_ticks | validation | `  ── Tick <T> [VAL END-ZONE] ── remaining=<R> \| airborne=<A> \| returned=<X>/<N> \| terminated=<B> \| truncated=<B>` | DEBUG |
| [train_full.py:888](train_full.py:888) | Validation END-ZONE per-aircraft dump | validation | `    <name> (id=<ID8>..): pos=(<lat>,<lon>) fuel=<F> rtb=<B> route_pts=<P>` | DEBUG |
| [train_full.py:895](train_full.py:895) | Validation: all agents RTB | validation | `  Validation: all agents RTB at tick <T>` | DEBUG |
| [train_full.py:898](train_full.py:898) | Validation env terminated/truncated | validation | `  Validation ended at tick <T>: terminated=<B>, truncated=<B>` | WARNING |
| [train_full.py:911](train_full.py:911) | Start of validation audit block | validation | `  --- Validation audit ---` | INFO |
| [train_full.py:925](train_full.py:925) | Per target, in validation audit | validation | `    t=<short> reach=[<a1,a2>] plan=[<b1>] hit=Y/N cheapest=<aid>:<cost>` | INFO |
| [train_full.py:935](train_full.py:935) | Per agent, in validation audit | validation | `    agent=<ID4> budget=<B> cap=<C> used=<U>/<C> plan=[<t1,t2>]` | INFO |
| [train_full.py:940](train_full.py:940) | Validation audit headline | validation | `  Hit: plan=<H>/<T> reachable=<R>/<RT> unreachable=<U>/<UT> dropped_reachable=<D> oracle_violations=<V>` | INFO |
| [train_full.py:947](train_full.py:947) | If audit found dropped reachable targets | validation | `  Dropped reachable targets (oracle chose not to plan): [<t1>,<t2>]` | INFO |
| [train_full.py:952](train_full.py:952) | If audit found unreachable targets attacked | error | `  ANOMALY: unreachable target(s) attacked: [<id>,<id>]` | ERROR |
| [train_full.py:958](train_full.py:958) | If oracle plan items were not executed | error | `  Oracle plan incomplete in execution — missed: [<t1>,..]` | WARNING |
| [train_full.py:967](train_full.py:967) | After validation `export_recording()` | validation | `  Validation recording exported: ep<NNN>_validation` | DEBUG |
| [train_full.py:969](train_full.py:969) | Validation recording export raised | error | `  Failed to export validation recording: <e>` | WARNING |
| [train_full.py:1050](train_full.py:1050) | `train_episode`: no attacking agents in scenario | error | `No attacking agents found!` | ERROR |
| [train_full.py:1055](train_full.py:1055) | `train_episode`: no tasks generated | error | `No tasks generated!` | ERROR |
| [train_full.py:1092](train_full.py:1092) | Per-episode compact scenario summary | episode-init | `Scenario: <N> agents [<types>] \| Blue base: (<lat>, <lon>)` | DEBUG |
| [train_full.py:1095](train_full.py:1095) | Same — targets line | episode-init | `  Targets (<N>): <name1>, <name2>, ...` | DEBUG |
| [train_full.py:1102-1114](train_full.py:1102) | `--verbose` only — AGENTS dump (id, location, budget, weapon, base, capabilities) | episode-init | `==... AGENTS ==... Agent <i>: <id> ...` (multi-line) | DEBUG, gated by `verbose=True` |
| [train_full.py:1116-1128](train_full.py:1116) | `--verbose` only — ALL TASKS dump | episode-init | `==... ALL TASKS (<N> total) ==... Task <i>: Target ID/Utility/Location/Action` | DEBUG |
| [train_full.py:1150-1163](train_full.py:1150) | `--verbose` only — TASK SPLIT dump | episode-init | `==... TASK SPLIT ==... Partial tasks (<N>): ... Full tasks ... Hidden targets: <set>` | DEBUG |
| [train_full.py:1167-1170](train_full.py:1167) | `--verbose` only — MATCH-AOU SOLUTIONS banner | episode-init | `==... MATCH-AOU SOLUTIONS ==...` | DEBUG |
| [train_full.py:1172](train_full.py:1172) | Before partial solve | episode-init | `Solving MATCH-AOU (partial)...` | DEBUG |
| [train_full.py:1178-1179](train_full.py:1178) | `--verbose` only — partial solution dump | episode-init | `  --- Partial Solution ---` then `_log_solution_details` lines | DEBUG |
| [train_full.py:1181](train_full.py:1181) | Before full/oracle solve | episode-init | `Solving MATCH-AOU (full / oracle)...` | DEBUG |
| [train_full.py:1187-1197](train_full.py:1187) | `--verbose` only — full solution + comparison dump | episode-init | `--- Full (Oracle) Solution --- ... --- Comparison --- Targets in partial: ...` | DEBUG |
| [train_full.py:1200](train_full.py:1200) | Partial solution empty → episode skipped | error | `Partial solution empty, skipping episode` | WARNING |
| [train_full.py:1223-1226](train_full.py:1223) | `--verbose` only — PRE-LAUNCH banner | episode-init | `==... PRE-LAUNCH ==...` | DEBUG |
| [train_full.py:1237](train_full.py:1237) | Per aircraft launched in pre-launch | episode-init | `  LAUNCH: <name> (id=<ID8>..) from airbase <ID8>..` | DEBUG |
| [train_full.py:1246](train_full.py:1246) | After all pre-launches | episode-init | `  Airborne after launch: <N> aircraft — [<names>]` | DEBUG |
| [train_full.py:1258-1269](train_full.py:1258) | `--verbose` only — EXECUTOR QUEUE dump | episode-init | `==... EXECUTOR QUEUE ==... Agent <id>: <Q> assignments  task=<i>, step=<j>, level=<L>, target=<T>` | DEBUG |
| [train_full.py:1287-1297](train_full.py:1287) | `--verbose` only — ORACLE SETUP dump | episode-init | `==... ORACLE SETUP ==... Partial target IDs (known): ... Full targets for <id>: ... → Agent should learn to attack: ...` | DEBUG |
| [train_full.py:1298-1300](train_full.py:1298) | `--verbose` only — SIMULATION START banner | episode-init | `==... SIMULATION START ==...` | DEBUG |
| [train_full.py:1321-1323](train_full.py:1321) | `--verbose` only — utility setup info | episode-init | `  Utility map: {...} \| Max utility: <U> \| Oracle total utility: <O>` | DEBUG (3 lines) |
| [train_full.py:1334](train_full.py:1334) | `n_agents > MAX_AGENTS` | error | `Scenario has <N> agents but MAX_AGENTS=<M>. Only the first <M> will be used for the critic.` | WARNING |
| [train_full.py:1351](train_full.py:1351) | Executor `next_action` raised inside the simulation loop | error | `Tick <T>: Executor error (skipping): <e>` | DEBUG |
| [train_full.py:1400](train_full.py:1400) | `build_observation_vector` raised on a tick | error | `Tick <T>: Can't observe <agent_id>: <e>` | DEBUG |
| [train_full.py:1412](train_full.py:1412) | Per (agent, target) discovery on a scan tick | mid-episode | `  Tick <T> DISCOVERY: agent <ID8>.. sees target <ID8>..` | DEBUG |
| [train_full.py:1497](train_full.py:1497) | Per RL decision (event-driven, rare) | mid-episode | `  Tick <T> RL DECISION: <ID8>.. \| trigger=<R> \| RL=<A> Oracle=<A> Match=✓/✗ Reward=<R> (rl_u=<U>, oracle_u=<U>)` | DEBUG |
| [train_full.py:1518](train_full.py:1518) | RL action invalid for `plan_edit_to_blade_action` | error | `  RL action <A> invalid for <agent_id>: <e>` | DEBUG |
| [train_full.py:1555](train_full.py:1555) | RL loop: agent transitions airborne→landed | mid-episode | `  Tick <T> RTB:     agent <ID8>.. landed` | DEBUG |
| [train_full.py:1564](train_full.py:1564) | RL END-ZONE block (every 10th tick of last 100) | mid-episode | `  ── Tick <T> [END-ZONE] ── remaining=<R> \| airborne=<A> \| returned=<X>/<N> \| terminated=<B> \| truncated=<B>` | DEBUG |
| [train_full.py:1579](train_full.py:1579) | END-ZONE per-aircraft dump | mid-episode | `    <name> (id=<ID8>..): pos=(<lat>,<lon>) fuel=<F> rtb=<B> route_pts=<P>` | DEBUG |
| [train_full.py:1587](train_full.py:1587) | All agents returned to base, ending episode | episode-end | `  All agents returned to base at tick <T> — ending episode` | DEBUG |
| [train_full.py:1592](train_full.py:1592) | Env terminated/truncated, ending episode | episode-end | `  Episode ended at tick <T>: terminated=<B>, truncated=<B> (env step count ≈ <S>)` | DEBUG |
| [train_full.py:1626](train_full.py:1626) | Episode-end utility summary | episode-end | `  Episode utility: achieved=<A> / oracle=<O> (ratio=<R>) → ep_reward=<X>` | DEBUG |
| [train_full.py:1643](train_full.py:1643) | After PPO update | training | `  PPO update: policy_loss=<P>, value_loss=<V>, entropy=<E>, clip_frac=<C>` | DEBUG |
| [train_full.py:1651](train_full.py:1651) | No transitions in buffer → PPO update skipped | error | `  No transitions collected, skipping PPO update` | WARNING |
| [train_full.py:1666](train_full.py:1666) | After RL `export_recording()` | episode-end | `  Recording exported: <label>` | DEBUG |
| [train_full.py:1668](train_full.py:1668) | RL recording export raised | error | `  Failed to export recording: <e>` | WARNING |
| [train_full.py:1745](train_full.py:1745) | `_log_solution_details` (called only when `--verbose`) | episode-init | `  Total assignments: <N>` | DEBUG |
| [train_full.py:1747](train_full.py:1747) | Same — per agent header | episode-init | `  Agent <id>:` | DEBUG |
| [train_full.py:1754](train_full.py:1754) | Same — per assignment | episode-init | `    task=<i> step=<j> level=<L> → target=<T>` | DEBUG |
| [train_full.py:2440](train_full.py:2440) | At startup, per old recording removed | run-init | `Removed old recording: <path>` | DEBUG |
| [train_full.py:2445](train_full.py:2445) | At startup, per old scenario removed | run-init | `Removed old scenario: <path>` | DEBUG |
| [train_full.py:2482-2503](train_full.py:2482) | Run-init banner block (~22 lines) | run-init | `=====...=====  Full RL Training — MAPPO + BLADE + MATCH-AOU  =====...=====  Base scenario: ...  Vary scenarios: ...  Episodes: ...` | INFO |
| [train_full.py:2522](train_full.py:2522) | Auto-computed time-feasibility cap (when `--vary-scenarios`) | run-init | `Time-feasibility cap: <K> km one-way (slowest=<class> <kmh> km/h, ticks=<T>, safety=<S>) [auto]` | INFO |
| [train_full.py:2530](train_full.py:2530) | Time-feasibility cap not computable | run-init | `Time-feasibility cap: not computed (empty pool)` | INFO |
| [train_full.py:2532](train_full.py:2532) | Manual override of time-feasibility cap | run-init | `Time-feasibility cap: <K> km one-way [manual override via --time-feasible-max-km]` | INFO |
| [train_full.py:2536](train_full.py:2536) | Scenario-generator setup line (when `--vary-scenarios`) | run-init | `ScenarioGenerator: aircraft_pool=[...], facility_pool=[...], aircraft=(min-max), facilities=(min-max), red_airbases=(min-max), max_dist=<K>km, vary_base=<B>` | INFO |
| [train_full.py:2546](train_full.py:2546) | BLADE setup banner | run-init | `--- Setting up BLADE environment ---` | INFO |
| [train_full.py:2552](train_full.py:2552) | RL components banner | run-init | `--- Creating RL components (MAPPO) ---` | INFO |
| [train_full.py:2566-2568](train_full.py:2566) | Network parameter count + shape | run-init | `ActorCriticNetwork: actor=<P> params, critic=<P> params  Actor: obs[<N>] → 128 → 64 → logits[<A>]  Critic: global[<N>] → 128 → 64 → V(s)[1]` (3 lines) | INFO |
| [train_full.py:2589](train_full.py:2589) | After PPOTrainer constructed | run-init | `PPOTrainer ready` | INFO |
| [train_full.py:2592-2594](train_full.py:2592) | "Starting Training" banner | run-init | `=====...===== Starting Training =====...=====` (3 lines) | INFO |
| [train_full.py:2609-2611](train_full.py:2609) | Per-episode separator banner | episode-init | `=====...===== Episode <N>/<T> =====...=====` (3 lines) | DEBUG |
| [train_full.py:2642](train_full.py:2642) | After scenario generation | episode-init | `  Generated scenario: <filename>` | DEBUG |
| [train_full.py:2703](train_full.py:2703) | Episode raised an exception | error | `!CRASH ep<NNNN>  <ExcType>: <msg>` | EXCEPTION (with traceback) |
| [train_full.py:2706](train_full.py:2706) | Same — follow-up banner | error | `!CRASH ep<NNNN>  (episode aborted, continuing)` | ERROR |
| [train_full.py:2732](train_full.py:2732) | Per-episode summary lines that contain a flag | summary | (one of the 2 compact summary lines) `[!FLAGS] ep<NNNN> [VAL ] ag=<N> tg=<N>[<E>e+<S>s]  L1:e=... s=...  L2:<outcome> split=<P>/<F>  ou=<U>/<U>` | WARNING (when flagged) |
| [train_full.py:2734](train_full.py:2734) | Per-episode summary lines that contain no flag | summary | same template, no `!FLAGS` prefix | INFO |
| [train_full.py:2753](train_full.py:2753) | Flagged episode triggering replay-for-recording | summary | `  → Replaying flagged episode for recording: <name>` | INFO (only fires when `should_record` was False, i.e. the cadence didn't already cover it) |
| [train_full.py:2771](train_full.py:2771) | Flagged-replay raised | error | `  Flagged-episode replay failed: <e>` | WARNING |
| [train_full.py:2799](train_full.py:2799) | Each line of progress block (5 lines) | training | (head) `========== Progress @ ep<NNNN> \| checkpoint saved \| rolling <W>ep ==========` then `Reward / Utility / Accuracy ... Ticks/ep / Actions / Decisions ... PPO loss / Flags(window): ...` then tail `=====...` | INFO |
| [train_full.py:2806](train_full.py:2806) | Checkpoint banner when progress-every is disabled | training | `=== Checkpoint saved (ep<NNNN>) \| rolling avg reward (last <N>): <R> ===` | INFO |
| [train_full.py:2813-2815](train_full.py:2813) | "Training Complete" banner | summary | `=====...===== Training Complete! =====...=====` (3 lines) | INFO |
| [train_full.py:2821-2824](train_full.py:2821) | Final metrics summary | summary | `Total episodes: <N>  Total PPO updates: <N>  Avg policy loss: <P>  Avg value loss: <V>` (4 lines) | INFO |
| [train_full.py:2830-2832](train_full.py:2830) | Last-10-episode aggregates | summary | `Avg reward (last 10): <R>  Avg accuracy (last 10): <A>%  Avg utility ratio (last 10): <U>%` (3 lines) | INFO |
| [train_full.py:2839](train_full.py:2839) | After `run_summary.write` | summary | `Run summary written to: <path>` | INFO |
| [train_full.py:2841](train_full.py:2841) | run_summary write raised | error | `Failed to write run_summary.txt: <e>` | WARNING |
| [train_full.py:2854](train_full.py:2854) | After `_write_highlights` | summary | `Highlights written to:  <path>` | INFO |
| [train_full.py:2856](train_full.py:2856) | highlights write raised | error | `Failed to write highlights.txt: <e>` | WARNING |
| [train_full.py:2858-2862](train_full.py:2858) | Output-paths summary | summary | `Outputs saved to: <dir>  Logs: <dir>/  Recordings: <dir>/  Models: <dir>/  Scenarios: <dir>/` (5 lines) | INFO |
| [src/match_aou/utils/blade_utils/scenario_generator.py:494](src/match_aou/utils/blade_utils/scenario_generator.py:494) | At `ScenarioGenerator.__init__` | run-init | `ScenarioGenerator ready: base=<basename>, aircraft_pool=[...], facility_pool=[...]` | INFO |
| [src/match_aou/utils/blade_utils/scenario_generator.py:625](src/match_aou/utils/blade_utils/scenario_generator.py:625) | When `include_sams=False` (each scenario gen) | episode-init | `  include_sams=False → removed all SAM facilities` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:678](src/match_aou/utils/blade_utils/scenario_generator.py:678) | Each unreachable target during reachability audit | episode-init | `Target '<name>' is unreachable by all agents - expected behavior for stretch targets` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:682](src/match_aou/utils/blade_utils/scenario_generator.py:682) | After reachability audit | episode-init | `Reachability audit: <R>/<T> targets reachable by at least one agent` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:768](src/match_aou/utils/blade_utils/scenario_generator.py:768) | Stretch zone collapsed by time-feasibility cap | episode-init | `  Stretch zone collapsed by time-feasibility cap (stretch_max=<X> ≤ stretch_min=<Y>)` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:776](src/match_aou/utils/blade_utils/scenario_generator.py:776) | When stretch zone is in use | episode-init | `  Target placement: <E> easy (≤<X>km), <S> stretch (<MIN>–<MAX>km)` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:784](src/match_aou/utils/blade_utils/scenario_generator.py:784) | When stretch disabled because fleet range too narrow | episode-init | `  Stretch targets disabled: fleet range gap (<G>km) too small for differentiation` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:834](src/match_aou/utils/blade_utils/scenario_generator.py:834) | A stretch target couldn't be placed and fell back to easy | episode-init | `  Stretch target fell back to easy zone` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:999](src/match_aou/utils/blade_utils/scenario_generator.py:999) | After discovery-chain pass | episode-init | `Discovery chain: easy relocated=<R>/<T> isolated=<I>, stretch relocated=<R>/<T> isolated=<I> (min fleet radar=<R> km)` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:1095](src/match_aou/utils/blade_utils/scenario_generator.py:1095) | A target could not be moved into in-zone radar reach | error | `Discovery chain: could not connect target '<name>' within zone bounds [<MIN>-<MAX> km]; leaving isolated` | WARNING |
| [src/match_aou/utils/blade_utils/scenario_generator.py:1298](src/match_aou/utils/blade_utils/scenario_generator.py:1298) | `_apply_fuel_tiers` — class not in tier map | episode-init | `  No fuel tier for class '<cls>'; keeping template fuel` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:1304](src/match_aou/utils/blade_utils/scenario_generator.py:1304) | `_apply_fuel_tiers` — invalid speed/fuelRate | episode-init | `  Cannot compute fuel for '<cls>' (speed/fuelRate invalid); keeping template fuel` | DEBUG |
| [src/match_aou/utils/blade_utils/scenario_generator.py:1343-1386](src/match_aou/utils/blade_utils/scenario_generator.py:1343) | `__main__` block of scenario_generator.py only | run-init | `Aircraft pool: [...]  Facility pool: [...]  Episode <N>: <path>  Aircraft (<C>): [...]  Facilities (<C>): [...]  RED airbases: <N>  Base: (<lat>, <lon>)` | print() — only fires when running `scenario_generator.py` directly, NOT during training |
| [src/match_aou/utils/blade_utils/scenario_factory.py:204](src/match_aou/utils/blade_utils/scenario_factory.py:204) | After `generate_all_enemy_tasks` | episode-init | `Generated <N> enemy tasks` | DEBUG |
| [src/match_aou/rl/training/fuel_damage.py:137](src/match_aou/rl/training/fuel_damage.py:137) | `plan_episode` — dice roll says no damage | episode-init | `Fuel damage: no damage this episode (dice roll)` | DEBUG (only when `--fuel-damage`) |
| [src/match_aou/rl/training/fuel_damage.py:161](src/match_aou/rl/training/fuel_damage.py:161) | Per scheduled fuel-damage event in `plan_episode` | episode-init | `  Fuel damage planned: agent=<ID8>.. tick=<T> factor=<F>` | DEBUG (only when `--fuel-damage`) |
| [src/match_aou/rl/training/fuel_damage.py:188](src/match_aou/rl/training/fuel_damage.py:188) | When a damage event activates mid-episode | mid-episode | `  *** FUEL DAMAGE at tick <T>: agent=<ID8>.. fuel reduced to <P>% ***` | DEBUG (only when `--fuel-damage`) |
| [src/match_aou/rl/training/ppo_trainer.py:244](src/match_aou/rl/training/ppo_trainer.py:244) | `update()` called with empty buffer | error | `Buffer empty, skipping update` | WARNING |
| [src/match_aou/rl/training/ppo_trainer.py:360](src/match_aou/rl/training/ppo_trainer.py:360) | Each `save_checkpoint` call | training | `Saved PPO checkpoint: <path>` | DEBUG |
| [src/match_aou/rl/training/ppo_trainer.py:370](src/match_aou/rl/training/ppo_trainer.py:370) | Each `load_checkpoint` call | training | `Loaded PPO checkpoint: <path> (episode <N>)` | INFO |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:197](src/match_aou/solvers/match_aou_MINLP_solver.py:197) | MINLP solver `solve()` returns non-optimal termination | error | `Model not solved to acceptable optimality. Check constraints and inputs.` | print() to stdout (no logging level) |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:223](src/match_aou/solvers/match_aou_MINLP_solver.py:223) | `display_solution` called with None | summary | `No solution found or problem is infeasible.` | print(); `display_solution` is not called from train_full.py |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:226](src/match_aou/solvers/match_aou_MINLP_solver.py:226) | `display_solution` header | summary | `Assigned Tasks:` | print() — same |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:228](src/match_aou/solvers/match_aou_MINLP_solver.py:228) | `display_solution` per agent | summary | `Agent <id> assigned to steps:` | print() — same |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:230](src/match_aou/solvers/match_aou_MINLP_solver.py:230) | `display_solution` per assignment | summary | `  Task <id>, Step <id>` | print() — same |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:232](src/match_aou/solvers/match_aou_MINLP_solver.py:232) | `display_solution` unassigned section | summary | `Unassigned Tasks:` | print() — same |
| [src/match_aou/solvers/match_aou_MINLP_solver.py:235](src/match_aou/solvers/match_aou_MINLP_solver.py:235) | `display_solution` per unassigned task | summary | `Task <j> is unassigned (Utility: <U>)` | print() — same |
| (vendored BLADE) `integrations/panopticon-main/gym/blade/utils/PlaybackRecorder.py:78` | After every `export_recording()` | episode-end | `Recording exported to '<path>'` | print() — fires for every `game.export_recording()` call from train_full.py (validation + RL) |
| (vendored BLADE) `integrations/panopticon-main/gym/blade/Game.py:799` | An exception inside Game | error | `<exception repr>` | print() — vendored BLADE; not refactored per CLAUDE.md §2 |

> Modules with logger handles but no observed call sites in active flows:
> [src/match_aou/rl/training/oracle.py:25](src/match_aou/rl/training/oracle.py:25)
> (lines 157, 163) and
> [src/match_aou/rl/training/episode_initializer.py:23](src/match_aou/rl/training/episode_initializer.py:23)
> (lines 68-245). Both are present in the package but **not imported by
> `train_full.py`** — they are dormant code. Their lines are listed in
> Section 5.

---

## 2. Disk output inventory

| Path pattern | Writer (file:line) | Cadence | Format | Purpose hint |
|---|---|---|---|---|
| `training_output/logs/training.log` | [train_full.py:2472](train_full.py:2472) (`logging.FileHandler`, mode `w`) | Once at startup; appended to throughout the run | Plain text, formatted as `%(asctime)s \| %(levelname)-7s \| %(name)s \| %(message)s` | Master mirror of console (same level: INFO normally, DEBUG with `--verbose`) |
| `training_output/logs/episode_NNNN.log` | [train_full.py:2657](train_full.py:2657) (per-episode `FileHandler`, mode `w`) | Attached at start of each episode; detached at end | Same plain-text formatter | Per-episode DEBUG firehose; one file per training episode |
| `training_output/logs/run_summary.txt` | [train_full.py:2327](train_full.py:2327) (`out_path.write_text`, called from [2838](train_full.py:2838)) | Once at end of training | Plain text (custom layout) | Flag counts, per-flag episode lists, cluster alerts, rolling-window aggregates, validation audit summary |
| `training_output/logs/highlights.txt` | [train_full.py:2194](train_full.py:2194) (`out_path.write_text`, called from [2848](train_full.py:2848)) | Once at end of training | Plain text (custom layout) | Curated index for Panopticon viewing — perfect-match eps, mismatch eps, learning-trend samples, flagged-ep index with recording filenames |
| `training_output/scenarios/episode_NNNN_scenario.json` | [src/match_aou/utils/blade_utils/scenario_generator.py:696-697](src/match_aou/utils/blade_utils/scenario_generator.py:696) (`open(out_path, "w") + json.dump`) | Every episode (when `--vary-scenarios`) | JSON (BLADE scenario schema, `indent=2`, `ensure_ascii=False`) | The varied scenario for that episode; reused by validation + RL phases |
| `training_output/recordings/ep<NNN>_validation Recording <start> - <end>.jsonl` | (vendored BLADE) `PlaybackRecorder.export_recording`, triggered by [train_full.py:966](train_full.py:966) | Every validation episode (cadence = `--validate-every`) | JSONL (one BLADE step per line; the BLADE `record_step` text buffer flushed verbatim) | Panopticon playback recording for the oracle-only validation run |
| `training_output/recordings/ep<NNN>_rl Recording <start> - <end>.jsonl` | (vendored BLADE) same; triggered by [train_full.py:1664](train_full.py:1664) | Every recordable training episode (cadence = `--record-every`) | JSONL | Panopticon playback recording for the RL-with-overrides run |
| `training_output/recordings/ep<NNNN>_flagged_<TAGS>_rl Recording <start> - <end>.jsonl` | (vendored BLADE) same; triggered by [train_full.py:1664](train_full.py:1664) under `record_name=replay_name` set at [train_full.py:2752](train_full.py:2752) | One per flagged episode that wasn't already covered by `--record-every` | JSONL | Replay recording for any episode that fired a !FLAG (TIMEOUT, ANOMALY, L2-fallback, L2-exhaust, noPPO) |
| `training_output/models/checkpoint_ep<N>.pt` | [src/match_aou/rl/training/ppo_trainer.py:345](src/match_aou/rl/training/ppo_trainer.py:345) (`torch.save`); call site [train_full.py:2777](train_full.py:2777) | Every `--save-freq` episodes | PyTorch checkpoint (`network_state`, `optimizer_state`, `episode_count`, `total_updates`, `metrics`, `config`) | Resumable training checkpoint |
| `training_output/models/final_model.pt` | [src/match_aou/rl/training/ppo_trainer.py:345](src/match_aou/rl/training/ppo_trainer.py:345) (`torch.save`); call site [train_full.py:2817](train_full.py:2817) | Once at end of training | PyTorch checkpoint (same fields as above) | Final trainer state |
| `training_output/models/actor_critic_final.pt` | [src/match_aou/rl/agent/network.py:249](src/match_aou/rl/agent/network.py:249) (`torch.save`); call site [train_full.py:2818](train_full.py:2818) | Once at end of training | PyTorch state-dict (`state_dict`, `obs_dim`, `action_dim`, `n_agents`, `hidden_size`) | Inference-only network weights |

> Note: each run starts by deleting all files under `recordings/` and
> `scenarios/` (and old per-episode `episode_*.log` files) — see
> [train_full.py:2438-2445](train_full.py:2438). `models/` is NOT cleaned —
> checkpoints accumulate across runs.

---

## 3. Runtime stdout sample

> Captured from a single 5-episode run (see "Runtime capture: how it was
> produced" above). Total 13,443 lines; included verbatim are the first 300
> and last 200, with the middle elided. Note: the `--verbose` flag pushes
> Pyomo's DEBUG messages onto the console too — most of the volume in the
> head is Pyomo model construction (`pyomo.core` / `pyomo.opt`), which is
> third-party, not project output. Unicode characters in the source code
> (e.g. `→`, `──`, `✓`, `✗`) appear escaped as `→` etc. because
> `sys.stdout` was not switched to UTF-8 in the captured shell.

```
2026-05-03 17:16:49,970 | INFO    | train_full | ======================================================================
2026-05-03 17:16:49,970 | INFO    | train_full | Full RL Training � MAPPO + BLADE + MATCH-AOU
2026-05-03 17:16:49,970 | INFO    | train_full | ======================================================================
2026-05-03 17:16:49,970 | INFO    | train_full | Base scenario:     data/scenarios/strike_training_4v5.json
2026-05-03 17:16:49,971 | INFO    | train_full | Vary scenarios:    True
2026-05-03 17:16:49,971 | INFO    | train_full | Episodes:          5
2026-05-03 17:16:49,971 | INFO    | train_full | RL trigger:        event-driven (discovery + fuel damage)
2026-05-03 17:16:49,971 | INFO    | train_full | Discovery scan:    every 50 ticks
2026-05-03 17:16:49,971 | INFO    | train_full | Max ticks:         14400
2026-05-03 17:16:49,971 | INFO    | train_full | Max agents:        5
2026-05-03 17:16:49,971 | INFO    | train_full | Learning rate:     0.0003
2026-05-03 17:16:49,971 | INFO    | train_full | Seed:              42
2026-05-03 17:16:49,971 | INFO    | train_full | Fuel damage:       True
2026-05-03 17:16:49,971 | INFO    | train_full | Include SAMs:      False
2026-05-03 17:16:49,972 | INFO    | train_full | Allowed aircraft:  all (from pool)
2026-05-03 17:16:49,972 | INFO    | train_full | Stretch ratio:     0.5
2026-05-03 17:16:49,972 | INFO    | train_full | Validate every:    1 episodes
2026-05-03 17:16:49,972 | INFO    | train_full | Record every:      1 episodes (0=never)
2026-05-03 17:16:49,972 | INFO    | train_full | Verbose console:   True
2026-05-03 17:16:49,972 | INFO    | train_full | DEBUG force flags: ['l2-fallback', 'timeout']
2026-05-03 17:16:49,973 | INFO    | train_full | Output dir:        C:\Users\Itama\PycharmProjects\Multi_Agent_Task_Allocation_and_Adaptation\training_output
2026-05-03 17:16:50,011 | INFO    | match_aou.utils.blade_utils.scenario_generator | ScenarioGenerator ready: base=strike_training_4v5.json, aircraft_pool=['B-2 Spirit', 'F-35A Lightning II', 'KC-135R Stratotanker', 'F-16 Fighting Falcon'], facility_pool=['Tor-M2', 'Pantsir-S1']
2026-05-03 17:16:50,012 | INFO    | train_full | Time-feasibility cap: 1195 km one-way (slowest=KC-135R Stratotanker 854 km/h, ticks=14400, safety=0.3) [auto]
2026-05-03 17:16:50,013 | INFO    | train_full | ScenarioGenerator: aircraft_pool=['B-2 Spirit', 'F-35A Lightning II', 'KC-135R Stratotanker', 'F-16 Fighting Falcon'], facility_pool=['Tor-M2', 'Pantsir-S1'], aircraft=(2-3), facilities=(2-4), red_airbases=(3-5), max_dist=2500.0km, vary_base=False
2026-05-03 17:16:50,013 | INFO    | train_full |
--- Setting up BLADE environment ---
2026-05-03 17:16:50,013 | INFO    | train_full | BLADE registered max_episode_steps: 2000
C:\Users\Itama\anaconda3\envs\nlp_env\Lib\site-packages\gymnasium\utils\passive_env_checker.py:158: UserWarning: [33mWARN: The obs returned by the `reset()` method is not within the observation space.[0m
  logger.warn(f"{pre} is not within the observation space.")
2026-05-03 17:16:50,022 | INFO    | train_full | BLADE env ready: duration=14400, max_episode_steps=14400, start_time=1699073110, current_time=1699073110
2026-05-03 17:16:50,023 | INFO    | train_full |
--- Creating RL components (MAPPO) ---
2026-05-03 17:16:50,037 | INFO    | train_full | ActorCriticNetwork: actor=12,549 params, critic=27,649 params
2026-05-03 17:16:50,037 | INFO    | train_full |   Actor:  obs[30] → 128 → 64 → logits[5]
2026-05-03 17:16:50,040 | INFO    | train_full |   Critic: global[150] → 128 → 64 → V(s)[1]
2026-05-03 17:16:53,967 | INFO    | train_full | PPOTrainer ready
2026-05-03 17:16:53,967 | INFO    | train_full |
======================================================================
2026-05-03 17:16:53,967 | INFO    | train_full | Starting Training
2026-05-03 17:16:53,968 | INFO    | train_full | ======================================================================
2026-05-03 17:16:53,968 | DEBUG   | train_full |
==================================================
2026-05-03 17:16:53,968 | DEBUG   | train_full | Episode 1/5
2026-05-03 17:16:53,968 | DEBUG   | train_full | ==================================================
2026-05-03 17:16:53,969 | DEBUG   | match_aou.utils.blade_utils.scenario_generator |   include_sams=False → removed all SAM facilities
2026-05-03 17:16:53,969 | DEBUG   | match_aou.utils.blade_utils.scenario_generator |   Stretch zone collapsed by time-feasibility cap (stretch_max=1195 ≤ stretch_min=1560)
2026-05-03 17:16:53,969 | DEBUG   | match_aou.utils.blade_utils.scenario_generator | Discovery chain: easy relocated=2/3 isolated=0, stretch relocated=0/0 isolated=0 (min fleet radar=93 km)
2026-05-03 17:16:53,969 | DEBUG   | match_aou.utils.blade_utils.scenario_generator | Reachability audit: 3/3 targets reachable by at least one agent
2026-05-03 17:16:54,005 | DEBUG   | train_full | Reloaded scenario from training_output\scenarios\episode_0000_scenario.json
2026-05-03 17:16:54,006 | DEBUG   | train_full |   Generated scenario: episode_0000_scenario.json
2026-05-03 17:16:54,009 | DEBUG   | train_full | --- Validation run (oracle only, no RL) ---
2026-05-03 17:16:54,011 | DEBUG   | match_aou.utils.blade_utils.scenario_factory | Generated 3 enemy tasks
2026-05-03 17:16:54,011 | DEBUG   | train_full | Validation: 2 agents, 3 tasks
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing ConcreteModel 'ConcreteModel', from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing SetOperator, name=A*T*S, from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing IndexedVar 'x' on [Model] from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing Variable x
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructed component ''[Model].x'':
x : Size=6, Index=A*T*S
    Key       : Lower : Value : Upper : Fixed : Stale : Domain
    (0, 0, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 1, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 2, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 0, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 1, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 2, 0) :     0 :  None :     1 : False :  True : Binary

2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing IndexedVar 'y' on [Model] from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing Variable y
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructed component ''[Model].y'':
y : Size=3, Index=T
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      0 :     0 :  None :     1 : False :  True : Binary
      1 :     0 :  None :     1 : False :  True : Binary
      2 :     0 :  None :     1 : False :  True : Binary

2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing ScalarObjective 'obj' on [Model] from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing objective obj
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructed component ''[Model].obj'':
obj : Size=1, Index=None, Active=True
    Key  : Active : Sense    : Expression
    None :   True : maximize : 80*y[0]*(1 - 1e-06**(x[0,0,0] + x[1,0,0])) + 80*y[1]*(1 - 1e-06**(x[0,1,0] + x[1,1,0])) + 80*y[2]*(1 - 1e-06**(x[0,2,0] + x[1,2,0]))

2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing SetOperator, name=A*T*S, from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'capability' on [Model] from data=None
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing constraint capability
2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructed component ''[Model].capability'':
capability : Size=0, Index=A*T*S, Active=True
    Key : Lower : Body : Upper : Active

2026-05-03 17:16:54,011 | DEBUG   | pyomo.core | Constructing ConstraintList 'dependency' on [Model] from data=None
2026-05-03 17:16:54,019 | DEBUG   | pyomo.core | Constructing constraint list dependency
2026-05-03 17:16:54,019 | DEBUG   | pyomo.core | Constructing Set, name={}, from data=None
2026-05-03 17:16:54,019 | DEBUG   | pyomo.core | Constructed component ''[Model].dependency'':
dependency : Size=0, Index={}, Active=True
    Key : Lower : Body : Upper : Active

2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing SetOperator, name=T*S, from data=None
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'task_step_allocation' on [Model] from data=None
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing constraint task_step_allocation
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructed component ''[Model].task_step_allocation'':
task_step_allocation : Size=3, Index=T*S, Active=True
    Key    : Lower : Body                         : Upper : Active
    (0, 0) :  -Inf : y[0] - (x[0,0,0] + x[1,0,0]) :   0.0 :   True
    (1, 0) :  -Inf : y[1] - (x[0,1,0] + x[1,1,0]) :   0.0 :   True
    (2, 0) :  -Inf : y[2] - (x[0,2,0] + x[1,2,0]) :   0.0 :   True

2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'task_full_allocation' on [Model] from data=None
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing constraint task_full_allocation
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructed component ''[Model].task_full_allocation'':
task_full_allocation : Size=3, Index=T, Active=True
    Key : Lower : Body                         : Upper : Active
      0 :  -Inf : x[0,0,0] + x[1,0,0] - 2*y[0] :   0.0 :   True
      1 :  -Inf : x[0,1,0] + x[1,1,0] - 2*y[1] :   0.0 :   True
      2 :  -Inf : x[0,2,0] + x[1,2,0] - 2*y[2] :   0.0 :   True

2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'movement_budget' on [Model] from data=None
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructing constraint movement_budget
2026-05-03 17:16:54,020 | DEBUG   | pyomo.core | Constructed component ''[Model].movement_budget'':
movement_budget : Size=2, Index=A, Active=True
    Key : Lower : Body                                                                                    : Upper              : Active
      0 :  -Inf : 21505.200645901266*x[0,0,0] + 22682.445747489983*x[0,1,0] + 23341.873812724036*x[0,2,0] :  60028.38357002235 :   True
      1 :  -Inf :  26356.69927317617*x[1,0,0] + 27799.526783800084*x[1,1,0] + 28607.719531863513*x[1,2,0] : 102998.80998674117 :   True

2026-05-03 17:16:57,700 | DEBUG   | pyomo.core | Writing model 'unknown' to file 'C:\Users\Itama\AppData\Local\Temp\tmpzo26z97v.pyomo.nl' with format nl
2026-05-03 17:16:57,748 | DEBUG   | pyomo.opt | Running ['C:\\Users\\Itama\\anaconda3\\envs\\nlp_env\\Library\\bin\\bonmin.exe', 'C:\\Users\\Itama\\AppData\\Local\\Temp\\tmpzo26z97v.pyomo.nl', '-AMPL']
2026-05-03 17:16:59,303 | DEBUG   | train_full |   → 5 assignments, 0 unselected
2026-05-03 17:16:59,303 | DEBUG   | train_full |   VAL plan: agent=be31019b → tasks=['e3626956', '6c6f7990']
2026-05-03 17:16:59,303 | DEBUG   | train_full |   VAL plan: agent=0a14f756 → tasks=['e3626956', '5880c13a', '6c6f7990']
C:\Users\Itama\anaconda3\envs\nlp_env\Lib\site-packages\gymnasium\utils\passive_env_checker.py:158: UserWarning: [33mWARN: The obs returned by the `step()` method is not within the observation space.[0m
  logger.warn(f"{pre} is not within the observation space.")
2026-05-03 17:16:59,327 | DEBUG   | train_full |   Validation LAUNCH: B-2 Spirit #698 (id=be31019b..) from airbase a3616929..
2026-05-03 17:16:59,328 | DEBUG   | train_full |   Validation LAUNCH: KC-135R Stratotanker #76 (id=0a14f756..) from airbase a3616929..
2026-05-03 17:16:59,329 | DEBUG   | train_full |   Tick     0 [VAL ] MOVE:   agent 0a14f756.. → (37.46175940933924, 38.749287831649916)
2026-05-03 17:16:59,329 | DEBUG   | train_full |   Tick     1 [VAL ] MOVE:   agent be31019b.. → (37.46175940933924, 38.749287831649916)
2026-05-03 17:17:01,445 | DEBUG   | train_full |   Tick  2140 [VAL ] ATTACK: agent be31019b.. → target e3626956..
2026-05-03 17:17:01,446 | DEBUG   | train_full |   Tick  2141 [VAL ] MOVE:   agent be31019b.. → (37.948591105129154, 38.999324415004665)
2026-05-03 17:17:01,631 | DEBUG   | train_full |   Tick  2340 [VAL ] ATTACK: agent be31019b.. → target 6c6f7990..
2026-05-03 17:17:01,633 | DEBUG   | train_full |   Tick  2341 [VAL ] RTB:    agent be31019b..
2026-05-03 17:17:01,920 | DEBUG   | train_full |   Tick  2622 [VAL ] ATTACK: agent 0a14f756.. → target e3626956..
2026-05-03 17:17:01,926 | DEBUG   | train_full |   Tick  2623 [VAL ] MOVE:   agent 0a14f756.. → (37.547035587823885, 39.3210685050754)
2026-05-03 17:17:02,095 | DEBUG   | train_full |   Tick  2805 [VAL ] ATTACK: agent 0a14f756.. → target 5880c13a..
2026-05-03 17:17:02,095 | DEBUG   | train_full |   Tick  2806 [VAL ] MOVE:   agent 0a14f756.. → (37.948591105129154, 38.999324415004665)
2026-05-03 17:17:02,196 | DEBUG   | train_full |   Tick  2903 [VAL ] ATTACK: agent 0a14f756.. → target 6c6f7990..
2026-05-03 17:17:02,200 | DEBUG   | train_full |   Tick  2904 [VAL ] RTB:    agent 0a14f756..
2026-05-03 17:17:03,602 | DEBUG   | train_full |   Tick  4432 VAL RTB: agent be31019b.. landed
2026-05-03 17:17:04,592 | DEBUG   | train_full |   Tick  5481 VAL RTB: agent 0a14f756.. landed
2026-05-03 17:17:04,594 | DEBUG   | train_full |   Validation: all agents RTB at tick 5481
2026-05-03 17:17:04,595 | INFO    | train_full |   --- Validation audit ---
2026-05-03 17:17:04,595 | INFO    | train_full |     t=e3626956 reach=[0a14,be31] plan=[be31,0a14] hit=Y cheapest=be31:21505
2026-05-03 17:17:04,595 | INFO    | train_full |     t=5880c13a reach=[0a14,be31] plan=[0a14] hit=Y cheapest=be31:22682
2026-05-03 17:17:04,596 | INFO    | train_full |     t=6c6f7990 reach=[0a14,be31] plan=[be31,0a14] hit=Y cheapest=be31:23342
2026-05-03 17:17:04,596 | INFO    | train_full |     agent=be31 budget=120057 cap=60028 used=44847/60028 plan=[e3626956,6c6f7990]
2026-05-03 17:17:04,596 | INFO    | train_full |     agent=0a14 budget=205998 cap=102999 used=82764/102999 plan=[e3626956,5880c13a,6c6f7990]
2026-05-03 17:17:04,597 | INFO    | train_full |   Hit: plan=3/3 reachable=3/3 unreachable=0/0 dropped_reachable=0 oracle_violations=0
2026-05-03 17:17:04,696 | DEBUG   | train_full |   Validation recording exported: ep001_validation
2026-05-03 17:17:04,698 | DEBUG   | train_full | Reloaded scenario from training_output\scenarios\episode_0000_scenario.json
2026-05-03 17:17:04,698 | DEBUG   | match_aou.utils.blade_utils.scenario_factory | Generated 3 enemy tasks
2026-05-03 17:17:04,698 | DEBUG   | train_full | Scenario: 2 agents ['B-2 Spirit', 'KC-135R Stratotanker'] | Blue base: (32.85, 35.31)
2026-05-03 17:17:04,698 | DEBUG   | train_full |   Targets (3): Red Airbase (37.46, 38.75), Red Airbase (37.55, 39.32), Red Airbase (37.95, 39.00)
2026-05-03 17:17:04,698 | DEBUG   | train_full |
2026-05-03 17:17:04,698 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,698 | DEBUG   | train_full | AGENTS
2026-05-03 17:17:04,698 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,698 | DEBUG   | train_full |   Agent 0: be31019b-75b4-4474-a90b-1b6249d735da
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Name:      (from scenario)
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Location:  (32.3542, 34.8124)
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Budget:    120057
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Weapon ID: 59a5a12e-a168-4a95-bcf3-8d14bd6fcea1
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Home base: a3616929-2446-4345-af5a-3a9986908c0d
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Capabilities: ['attack', 'attack', 'attack']
2026-05-03 17:17:04,698 | DEBUG   | train_full |   Agent 1: 0a14f756-13f2-4c78-8aa8-446da245aee5
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Name:      (from scenario)
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Location:  (32.3542, 34.8124)
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Budget:    205998
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Weapon ID: f9992d58-30b3-4156-a114-4a89e48ca2f1
2026-05-03 17:17:04,698 | DEBUG   | train_full |     Home base: a3616929-2446-4345-af5a-3a9986908c0d
2026-05-03 17:17:04,706 | DEBUG   | train_full |     Capabilities: ['attack', 'attack', 'attack']
2026-05-03 17:17:04,707 | DEBUG   | train_full |
2026-05-03 17:17:04,710 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,711 | DEBUG   | train_full | ALL TASKS (3 total)
2026-05-03 17:17:04,712 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,712 | DEBUG   | train_full |   Task 0:
2026-05-03 17:17:04,712 | DEBUG   | train_full |     Target ID: e3626956-04af-4440-990f-a1088445cc9b
2026-05-03 17:17:04,713 | DEBUG   | train_full |     Utility:   80
2026-05-03 17:17:04,714 | DEBUG   | train_full |     Location:  (37.4618, 38.7493)
2026-05-03 17:17:04,714 | DEBUG   | train_full |     Action:    handle_aircraft_attack('AGENT_ID', 'e3626956-04af-4440-990f-a1088445cc9b', 'WEAPON_ID', 2)
2026-05-03 17:17:04,714 | DEBUG   | train_full |   Task 1:
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Target ID: 5880c13a-bddf-473f-b10c-a58ebc806230
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Utility:   80
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Location:  (37.5470, 39.3211)
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Action:    handle_aircraft_attack('AGENT_ID', '5880c13a-bddf-473f-b10c-a58ebc806230', 'WEAPON_ID', 2)
2026-05-03 17:17:04,715 | DEBUG   | train_full |   Task 2:
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Target ID: 6c6f7990-d33c-495b-a60f-67c66f03253e
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Utility:   80
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Location:  (37.9486, 38.9993)
2026-05-03 17:17:04,715 | DEBUG   | train_full |     Action:    handle_aircraft_attack('AGENT_ID', '6c6f7990-d33c-495b-a60f-67c66f03253e', 'WEAPON_ID', 2)
2026-05-03 17:17:04,715 | DEBUG   | train_full | Discovery chain (split): clean (hidden=1, known=2, isolated_pinned=0, min_radar=93 km)
2026-05-03 17:17:04,715 | DEBUG   | train_full | Task split: 2 partial, 3 full, 1 hidden
2026-05-03 17:17:04,715 | DEBUG   | train_full |
2026-05-03 17:17:04,715 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,715 | DEBUG   | train_full | TASK SPLIT
2026-05-03 17:17:04,715 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,715 | DEBUG   | train_full |   Partial tasks (2):
2026-05-03 17:17:04,715 | DEBUG   | train_full |     [0] target=e3626956-04af-4440-990f-a1088445cc9b, utility=80
2026-05-03 17:17:04,715 | DEBUG   | train_full |     [1] target=5880c13a-bddf-473f-b10c-a58ebc806230, utility=80
2026-05-03 17:17:04,715 | DEBUG   | train_full |   Full tasks (3):
2026-05-03 17:17:04,715 | DEBUG   | train_full |     [0] target=e3626956-04af-4440-990f-a1088445cc9b, utility=80
2026-05-03 17:17:04,721 | DEBUG   | train_full |     [1] target=5880c13a-bddf-473f-b10c-a58ebc806230, utility=80
2026-05-03 17:17:04,722 | DEBUG   | train_full |     [2] target=6c6f7990-d33c-495b-a60f-67c66f03253e, utility=80 *** HIDDEN ***
2026-05-03 17:17:04,722 | DEBUG   | train_full |   Hidden targets: {'6c6f7990-d33c-495b-a60f-67c66f03253e'}
2026-05-03 17:17:04,722 | DEBUG   | train_full |
2026-05-03 17:17:04,722 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,729 | DEBUG   | train_full | MATCH-AOU SOLUTIONS
2026-05-03 17:17:04,730 | DEBUG   | train_full | ============================================================
2026-05-03 17:17:04,730 | DEBUG   | train_full | Solving MATCH-AOU (partial)...
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing ConcreteModel 'ConcreteModel', from data=None
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing SetOperator, name=A*T*S, from data=None
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing IndexedVar 'x' on [Model] from data=None
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructing Variable x
2026-05-03 17:17:04,730 | DEBUG   | pyomo.core | Constructed component ''[Model].x'':
x : Size=4, Index=A*T*S
    Key       : Lower : Value : Upper : Fixed : Stale : Domain
    (0, 0, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 1, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 0, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 1, 0) :     0 :  None :     1 : False :  True : Binary

2026-05-03 17:17:04,841 | DEBUG   | pyomo.core | Constructing IndexedVar 'y' on [Model] from data=None
2026-05-03 17:17:04,842 | DEBUG   | pyomo.core | Constructing Variable y
2026-05-03 17:17:04,843 | DEBUG   | pyomo.core | Constructed component ''[Model].y'':
y : Size=2, Index=T
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      0 :     0 :  None :     1 : False :  True : Binary
      1 :     0 :  None :     1 : False :  True : Binary

2026-05-03 17:17:04,844 | DEBUG   | pyomo.core | Constructing ScalarObjective 'obj' on [Model] from data=None
2026-05-03 17:17:04,844 | DEBUG   | pyomo.core | Constructing objective obj
2026-05-03 17:17:04,846 | DEBUG   | pyomo.core | Constructed component ''[Model].obj'':
obj : Size=1, Index=None, Active=True
    Key  : Active : Sense    : Expression
    None :   True : maximize : 80*y[0]*(1 - 1e-06**(x[0,0,0] + x[1,0,0])) + 80*y[1]*(1 - 1e-06**(x[0,1,0] + x[1,1,0]))

2026-05-03 17:17:04,847 | DEBUG   | pyomo.core | Constructing SetOperator, name=A*T*S, from data=None
2026-05-03 17:17:04,847 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'capability' on [Model] from data=None
2026-05-03 17:17:04,848 | DEBUG   | pyomo.core | Constructing constraint capability
2026-05-03 17:17:04,848 | DEBUG   | pyomo.core | Constructed component ''[Model].capability'':
capability : Size=0, Index=A*T*S, Active=True
    Key : Lower : Body : Upper : Active

2026-05-03 17:17:04,849 | DEBUG   | pyomo.core | Constructing ConstraintList 'dependency' on [Model] from data=None
2026-05-03 17:17:04,849 | DEBUG   | pyomo.core | Constructing constraint list dependency
2026-05-03 17:17:04,850 | DEBUG   | pyomo.core | Constructing Set, name={}, from data=None
2026-05-03 17:17:04,850 | DEBUG   | pyomo.core | Constructed component ''[Model].dependency'':
dependency : Size=0, Index={}, Active=True
    Key : Lower : Body : Upper : Active

2026-05-03 17:17:04,851 | DEBUG   | pyomo.core | Constructing SetOperator, name=T*S, from data=None
2026-05-03 17:17:04,852 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'task_step_allocation' on [Model] from data=None
2026-05-03 17:17:04,852 | DEBUG   | pyomo.core | Constructing constraint task_step_allocation
2026-05-03 17:17:04,854 | DEBUG   | pyomo.core | Constructed component ''[Model].task_step_allocation'':
task_step_allocation : Size=2, Index=T*S, Active=True
    Key    : Lower : Body                         : Upper : Active
    (0, 0) :  -Inf : y[0] - (x[0,0,0] + x[1,0,0]) :   0.0 :   True
    (1, 0) :  -Inf : y[1] - (x[0,1,0] + x[1,1,0]) :   0.0 :   True

2026-05-03 17:17:04,854 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'task_full_allocation' on [Model] from data=None
2026-05-03 17:17:04,855 | DEBUG   | pyomo.core | Constructing constraint task_full_allocation
2026-05-03 17:17:04,860 | DEBUG   | pyomo.core | Constructed component ''[Model].task_full_allocation'':
task_full_allocation : Size=2, Index=T, Active=True
    Key : Lower : Body                         : Upper : Active
      0 :  -Inf : x[0,0,0] + x[1,0,0] - 2*y[0] :   0.0 :   True
      1 :  -Inf : x[0,1,0] + x[1,1,0] - 2*y[1] :   0.0 :   True

2026-05-03 17:17:04,861 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'movement_budget' on [Model] from data=None
2026-05-03 17:17:04,861 | DEBUG   | pyomo.core | Constructing constraint movement_budget
2026-05-03 17:17:04,863 | DEBUG   | pyomo.core | Constructed component ''[Model].movement_budget'':
movement_budget : Size=2, Index=A, Active=True
    Key : Lower : Body                                                      : Upper              : Active
      0 :  -Inf : 21505.200645901266*x[0,0,0] + 22682.445747489983*x[0,1,0] :  60028.38357002235 :   True
      1 :  -Inf :  26356.69927317617*x[1,0,0] + 27799.526783800084*x[1,1,0] : 102998.80998674117 :   True

2026-05-03 17:17:05,026 | DEBUG   | pyomo.core | Writing model 'unknown' to file 'C:\Users\Itama\AppData\Local\Temp\tmp0ethip7b.pyomo.nl' with format nl
2026-05-03 17:17:05,033 | DEBUG   | pyomo.opt | Running ['C:\\Users\\Itama\\anaconda3\\envs\\nlp_env\\Library\\bin\\bonmin.exe', 'C:\\Users\\Itama\\AppData\\Local\\Temp\\tmp0ethip7b.pyomo.nl', '-AMPL']
Recording exported to 'training_output\recordings/ep001_validation Recording 064510 - 081649.jsonl'
2026-05-03 17:17:06,331 | DEBUG   | train_full |   → 4 assignments, 0 unselected
2026-05-03 17:17:06,331 | DEBUG   | train_full |   --- Partial Solution ---
2026-05-03 17:17:06,331 | DEBUG   | train_full |   Total assignments: 4
2026-05-03 17:17:06,331 | DEBUG   | train_full |   Agent be31019b-75b4-4474-a90b-1b6249d735da:
2026-05-03 17:17:06,331 | DEBUG   | train_full |     task=0 step=0 level=0 → target=e3626956-04af-4440-990f-a1088445cc9b
2026-05-03 17:17:06,331 | DEBUG   | train_full |     task=1 step=0 level=0 → target=5880c13a-bddf-473f-b10c-a58ebc806230
2026-05-03 17:17:06,331 | DEBUG   | train_full |   Agent 0a14f756-13f2-4c78-8aa8-446da245aee5:
2026-05-03 17:17:06,331 | DEBUG   | train_full |     task=0 step=0 level=0 → target=e3626956-04af-4440-990f-a1088445cc9b
2026-05-03 17:17:06,331 | DEBUG   | train_full |     task=1 step=0 level=0 → target=5880c13a-bddf-473f-b10c-a58ebc806230

[... 12943 lines elided ...]

2026-05-03 17:18:26,179 | DEBUG   | train_full |   Agent be31019b-75b4-4474-a90b-1b6249d735da:
2026-05-03 17:18:26,179 | DEBUG   | train_full |     task=0 step=0 level=0 → target=e3626956-04af-4440-990f-a1088445cc9b
2026-05-03 17:18:26,180 | DEBUG   | train_full |     task=1 step=0 level=0 → target=682a7461-054f-4d6e-adb4-4ee288f4d7ac
2026-05-03 17:18:26,180 | DEBUG   | train_full | Solving MATCH-AOU (full / oracle)...
2026-05-03 17:18:26,180 | DEBUG   | pyomo.core | Constructing ConcreteModel 'ConcreteModel', from data=None
2026-05-03 17:18:26,181 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:18:26,181 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:18:26,182 | DEBUG   | pyomo.core | Constructing RangeSet, name=FiniteScalarRangeSet, from data=None
2026-05-03 17:18:26,182 | DEBUG   | pyomo.core | Constructing SetOperator, name=A*T*S, from data=None
2026-05-03 17:18:26,182 | DEBUG   | pyomo.core | Constructing IndexedVar 'x' on [Model] from data=None
2026-05-03 17:18:26,183 | DEBUG   | pyomo.core | Constructing Variable x
2026-05-03 17:18:26,184 | DEBUG   | pyomo.core | Constructed component ''[Model].x'':
x : Size=10, Index=A*T*S
    Key       : Lower : Value : Upper : Fixed : Stale : Domain
    (0, 0, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 1, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 2, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 3, 0) :     0 :  None :     1 : False :  True : Binary
    (0, 4, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 0, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 1, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 2, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 3, 0) :     0 :  None :     1 : False :  True : Binary
    (1, 4, 0) :     0 :  None :     1 : False :  True : Binary

2026-05-03 17:18:26,184 | DEBUG   | pyomo.core | Constructing IndexedVar 'y' on [Model] from data=None
2026-05-03 17:18:26,184 | DEBUG   | pyomo.core | Constructing Variable y
2026-05-03 17:18:26,185 | DEBUG   | pyomo.core | Constructed component ''[Model].y'':
y : Size=5, Index=T
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      0 :     0 :  None :     1 : False :  True : Binary
      1 :     0 :  None :     1 : False :  True : Binary
      2 :     0 :  None :     1 : False :  True : Binary
      3 :     0 :  None :     1 : False :  True : Binary
      4 :     0 :  None :     1 : False :  True : Binary

2026-05-03 17:18:26,186 | DEBUG   | pyomo.core | Constructing ScalarObjective 'obj' on [Model] from data=None
2026-05-03 17:18:26,186 | DEBUG   | pyomo.core | Constructing objective obj
2026-05-03 17:18:26,187 | DEBUG   | pyomo.core | Constructed component ''[Model].obj'':
obj : Size=1, Index=None, Active=True
    Key  : Active : Sense    : Expression
    None :   True : maximize : 80*y[0]*(1 - 1e-06**(x[0,0,0] + x[1,0,0])) + 80*y[1]*(1 - 1e-06**(x[0,1,0] + x[1,1,0])) + 80*y[2]*(1 - 1e-06**(x[0,2,0] + x[1,2,0])) + 80*y[3]*(1 - 1e-06**(x[0,3,0] + x[1,3,0])) + 80*y[4]*(1 - 1e-06**(x[0,4,0] + x[1,4,0]))

2026-05-03 17:18:26,188 | DEBUG   | pyomo.core | Constructing SetOperator, name=A*T*S, from data=None
2026-05-03 17:18:26,188 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'capability' on [Model] from data=None
2026-05-03 17:18:26,188 | DEBUG   | pyomo.core | Constructing constraint capability
2026-05-03 17:18:26,189 | DEBUG   | pyomo.core | Constructed component ''[Model].capability'':
capability : Size=0, Index=A*T*S, Active=True
    Key : Lower : Body : Upper : Active

2026-05-03 17:18:26,189 | DEBUG   | pyomo.core | Constructing ConstraintList 'dependency' on [Model] from data=None
2026-05-03 17:18:26,190 | DEBUG   | pyomo.core | Constructing constraint list dependency
2026-05-03 17:18:26,190 | DEBUG   | pyomo.core | Constructing Set, name={}, from data=None
2026-05-03 17:18:26,190 | DEBUG   | pyomo.core | Constructed component ''[Model].dependency'':
dependency : Size=0, Index={}, Active=True
    Key : Lower : Body : Upper : Active

2026-05-03 17:18:26,190 | DEBUG   | pyomo.core | Constructing SetOperator, name=T*S, from data=None
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'task_step_allocation' on [Model] from data=None
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructing constraint task_step_allocation
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructed component ''[Model].task_step_allocation'':
task_step_allocation : Size=5, Index=T*S, Active=True
    Key    : Lower : Body                         : Upper : Active
    (0, 0) :  -Inf : y[0] - (x[0,0,0] + x[1,0,0]) :   0.0 :   True
    (1, 0) :  -Inf : y[1] - (x[0,1,0] + x[1,1,0]) :   0.0 :   True
    (2, 0) :  -Inf : y[2] - (x[0,2,0] + x[1,2,0]) :   0.0 :   True
    (3, 0) :  -Inf : y[3] - (x[0,3,0] + x[1,3,0]) :   0.0 :   True
    (4, 0) :  -Inf : y[4] - (x[0,4,0] + x[1,4,0]) :   0.0 :   True

2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'task_full_allocation' on [Model] from data=None
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructing constraint task_full_allocation
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructed component ''[Model].task_full_allocation'':
task_full_allocation : Size=5, Index=T, Active=True
    Key : Lower : Body                         : Upper : Active
      0 :  -Inf : x[0,0,0] + x[1,0,0] - 2*y[0] :   0.0 :   True
      1 :  -Inf : x[0,1,0] + x[1,1,0] - 2*y[1] :   0.0 :   True
      2 :  -Inf : x[0,2,0] + x[1,2,0] - 2*y[2] :   0.0 :   True
      3 :  -Inf : x[0,3,0] + x[1,3,0] - 2*y[3] :   0.0 :   True
      4 :  -Inf : x[0,4,0] + x[1,4,0] - 2*y[4] :   0.0 :   True

2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructing IndexedConstraint 'movement_budget' on [Model] from data=None
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructing constraint movement_budget
2026-05-03 17:18:26,191 | DEBUG   | pyomo.core | Constructed component ''[Model].movement_budget'':
movement_budget : Size=2, Index=A, Active=True
    Key : Lower : Body                                                                                                                                                : Upper              : Active
      0 :  -Inf : 3382.3350012753385*x[0,0,0] + 1154.7444918862134*x[0,1,0] + 3420.1210245394727*x[0,2,0] + 1184.5314726851132*x[0,3,0] + 1156.0325697201204*x[0,4,0] : 1388.2235545484832 :   True
      1 :  -Inf :   39001.61510320147*x[1,0,0] + 13315.328079006513*x[1,1,0] + 39437.32473429144*x[1,2,0] + 13658.800963793827*x[1,3,0] + 13330.180870312539*x[1,4,0] :  60028.38357002235 :   True

2026-05-03 17:18:26,234 | DEBUG   | pyomo.core | Writing model 'unknown' to file 'C:\Users\Itama\AppData\Local\Temp\tmp8h2e29bq.pyomo.nl' with format nl
2026-05-03 17:18:26,240 | DEBUG   | pyomo.opt | Running ['C:\\Users\\Itama\\anaconda3\\envs\\nlp_env\\Library\\bin\\bonmin.exe', 'C:\\Users\\Itama\\AppData\\Local\\Temp\\tmp8h2e29bq.pyomo.nl', '-AMPL']
2026-05-03 17:18:29,477 | DEBUG   | train_full |   → 4 assignments, 2 unselected
2026-05-03 17:18:29,477 | DEBUG   | train_full |   --- Full (Oracle) Solution ---
2026-05-03 17:18:29,478 | DEBUG   | train_full |   Total assignments: 4
2026-05-03 17:18:29,478 | DEBUG   | train_full |   Agent a80f591e-3488-4fea-83d1-f42bced92a72:
2026-05-03 17:18:29,478 | DEBUG   | train_full |     task=0 step=0 level=0 → target=5880c13a-bddf-473f-b10c-a58ebc806230
2026-05-03 17:18:29,478 | DEBUG   | train_full |   Agent be31019b-75b4-4474-a90b-1b6249d735da:
2026-05-03 17:18:29,478 | DEBUG   | train_full |     task=0 step=0 level=0 → target=5880c13a-bddf-473f-b10c-a58ebc806230
2026-05-03 17:18:29,478 | DEBUG   | train_full |     task=1 step=0 level=0 → target=7d0abcce-d8c1-47d1-a293-147a465982e7
2026-05-03 17:18:29,479 | DEBUG   | train_full |     task=2 step=0 level=0 → target=682a7461-054f-4d6e-adb4-4ee288f4d7ac
2026-05-03 17:18:29,479 | DEBUG   | train_full |   --- Comparison ---
2026-05-03 17:18:29,479 | DEBUG   | train_full |   Targets in partial: {'e3626956-04af-4440-990f-a1088445cc9b', '682a7461-054f-4d6e-adb4-4ee288f4d7ac'}
2026-05-03 17:18:29,479 | DEBUG   | train_full |   Targets in full:    {'7d0abcce-d8c1-47d1-a293-147a465982e7', '5880c13a-bddf-473f-b10c-a58ebc806230', '682a7461-054f-4d6e-adb4-4ee288f4d7ac'}
2026-05-03 17:18:29,479 | DEBUG   | train_full |   NEW in full (what RL should learn to attack): {'7d0abcce-d8c1-47d1-a293-147a465982e7', '5880c13a-bddf-473f-b10c-a58ebc806230'}
2026-05-03 17:18:29,490 | DEBUG   | train_full |
2026-05-03 17:18:29,490 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,490 | DEBUG   | train_full | PRE-LAUNCH
2026-05-03 17:18:29,490 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,490 | DEBUG   | train_full |   LAUNCH: F-16 Fighting Falcon #477 (id=a80f591e..) from airbase a3616929..
2026-05-03 17:18:29,490 | DEBUG   | train_full |   LAUNCH: B-2 Spirit #698 (id=be31019b..) from airbase a3616929..
2026-05-03 17:18:29,495 | DEBUG   | train_full |   Airborne after launch: 2 aircraft � ['F-16 Fighting Falcon #477', 'B-2 Spirit #698']
2026-05-03 17:18:29,495 | DEBUG   | train_full |
2026-05-03 17:18:29,495 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,495 | DEBUG   | train_full | EXECUTOR QUEUE
2026-05-03 17:18:29,495 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,495 | DEBUG   | train_full |   Agent a80f591e-3488-4fea-83d1-f42bced92a72: 1 assignments
2026-05-03 17:18:29,495 | DEBUG   | train_full |     task=1, step=0, level=0, target=682a7461-054f-4d6e-adb4-4ee288f4d7ac
2026-05-03 17:18:29,495 | DEBUG   | train_full |   Agent be31019b-75b4-4474-a90b-1b6249d735da: 2 assignments
2026-05-03 17:18:29,495 | DEBUG   | train_full |     task=0, step=0, level=0, target=e3626956-04af-4440-990f-a1088445cc9b
2026-05-03 17:18:29,495 | DEBUG   | train_full |     task=1, step=0, level=0, target=682a7461-054f-4d6e-adb4-4ee288f4d7ac
2026-05-03 17:18:29,495 | DEBUG   | train_full |
2026-05-03 17:18:29,495 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,495 | DEBUG   | train_full | ORACLE SETUP
2026-05-03 17:18:29,495 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,495 | DEBUG   | train_full |   Partial target IDs (known): {'e3626956-04af-4440-990f-a1088445cc9b', '682a7461-054f-4d6e-adb4-4ee288f4d7ac'}
2026-05-03 17:18:29,495 | DEBUG   | train_full |   Full targets for a80f591e-3488-4fea-83d1-f42bced92a72: {'5880c13a-bddf-473f-b10c-a58ebc806230'}
2026-05-03 17:18:29,495 | DEBUG   | train_full |     → Agent should learn to attack: {'5880c13a-bddf-473f-b10c-a58ebc806230'}
2026-05-03 17:18:29,495 | DEBUG   | train_full |   Full targets for be31019b-75b4-4474-a90b-1b6249d735da: {'7d0abcce-d8c1-47d1-a293-147a465982e7', '5880c13a-bddf-473f-b10c-a58ebc806230', '682a7461-054f-4d6e-adb4-4ee288f4d7ac'}
2026-05-03 17:18:29,495 | DEBUG   | train_full |     → Agent should learn to attack: {'7d0abcce-d8c1-47d1-a293-147a465982e7', '5880c13a-bddf-473f-b10c-a58ebc806230'}
2026-05-03 17:18:29,495 | DEBUG   | train_full |
2026-05-03 17:18:29,495 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,495 | DEBUG   | train_full | SIMULATION START
2026-05-03 17:18:29,502 | DEBUG   | train_full | ============================================================
2026-05-03 17:18:29,502 | DEBUG   | train_full |   Utility map: {'e3626956-04af-4440-990f-a1088445cc9b': 80, '5880c13a-bddf-473f-b10c-a58ebc806230': 80, '6c6f7990-d33c-495b-a60f-67c66f03253e': 80, '7d0abcce-d8c1-47d1-a293-147a465982e7': 80, '682a7461-054f-4d6e-adb4-4ee288f4d7ac': 80}
2026-05-03 17:18:29,503 | DEBUG   | train_full |   Max utility: 80
2026-05-03 17:18:29,503 | DEBUG   | train_full |   Oracle total utility: 240.0
2026-05-03 17:18:29,503 | DEBUG   | match_aou.rl.training.fuel_damage |   Fuel damage planned: agent=be31019b.. tick=6802 factor=0.23
2026-05-03 17:18:29,505 | DEBUG   | train_full |   Tick     0 [EXEC] MOVE:   agent a80f591e.. → (36.087239432240665, 34.458629269187384)
2026-05-03 17:18:29,539 | DEBUG   | train_full |   Tick     1 [EXEC] MOVE:   agent be31019b.. → (41.76087709186531, 41.87319123752358)
2026-05-03 17:18:29,738 | DEBUG   | train_full |   Tick   350 DISCOVERY: agent a80f591e.. sees target 5880c13a..
2026-05-03 17:18:29,741 | DEBUG   | train_full |   Tick   350 RL DECISION: a80f591e.. | trigger=discovery | RL=ATTACK_1 Oracle=ATTACK_1 Match=✓ Reward=+1.00 (rl_u=80, oracle_u=80)
2026-05-03 17:18:29,742 | DEBUG   | train_full |   Tick   350 [RL  ] ATTACK: agent a80f591e.. → target 5880c13a..
2026-05-03 17:18:29,770 | DEBUG   | train_full |   Tick   400 DISCOVERY: agent a80f591e.. sees target 7d0abcce..
2026-05-03 17:18:29,805 | DEBUG   | train_full |   Tick   400 RL DECISION: a80f591e.. | trigger=discovery | RL=ATTACK_1 Oracle=ATTACK_1 Match=✓ Reward=+1.00 (rl_u=80, oracle_u=80)
2026-05-03 17:18:29,805 | DEBUG   | train_full |   Tick   400 [RL  ] ATTACK: agent a80f591e.. → target 5880c13a..
2026-05-03 17:18:29,885 | DEBUG   | train_full |   Tick   547 [EXEC] ATTACK: agent a80f591e.. → target 682a7461..
2026-05-03 17:18:29,887 | DEBUG   | train_full |   Tick   548 [EXEC] RTB:    agent a80f591e..
2026-05-03 17:18:30,090 | DEBUG   | train_full |   ── Tick  1000 ── airborne: 2/2 | RL decisions: 2 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:30,098 | DEBUG   | train_full |   Tick  1024 RTB:     agent a80f591e.. landed
2026-05-03 17:18:30,490 | DEBUG   | train_full |   ── Tick  2000 ── airborne: 1/2 | RL decisions: 2 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:30,939 | DEBUG   | train_full |   ── Tick  3000 ── airborne: 1/2 | RL decisions: 2 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:31,405 | DEBUG   | train_full |   Tick  3950 DISCOVERY: agent be31019b.. sees target 6c6f7990..
2026-05-03 17:18:31,407 | DEBUG   | train_full |   Tick  3950 RL DECISION: be31019b.. | trigger=discovery | RL=RTB Oracle=NOOP Match=✗ Reward=+0.00 (rl_u=0, oracle_u=0)
2026-05-03 17:18:31,407 | DEBUG   | train_full |   Tick  3950 [RL  ] RTB:    agent be31019b..
2026-05-03 17:18:31,461 | DEBUG   | train_full |   ── Tick  4000 ── airborne: 1/2 | RL decisions: 3 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:32,008 | DEBUG   | train_full |   ── Tick  5000 ── airborne: 1/2 | RL decisions: 3 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:32,568 | DEBUG   | train_full |   ── Tick  6000 ── airborne: 1/2 | RL decisions: 3 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:33,195 | DEBUG   | match_aou.rl.training.fuel_damage |   *** FUEL DAMAGE at tick 6802: agent=be31019b.. fuel reduced to 23% ***
2026-05-03 17:18:33,225 | DEBUG   | train_full |   Tick  6802 RL DECISION: be31019b.. | trigger=fuel_damage | RL=RTB Oracle=NOOP Match=✗ Reward=+0.00 (rl_u=0, oracle_u=0)
2026-05-03 17:18:33,225 | DEBUG   | train_full |   Tick  6802 [RL  ] RTB:    agent be31019b..
2026-05-03 17:18:33,328 | DEBUG   | train_full |   ── Tick  7000 ── airborne: 1/2 | RL decisions: 4 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:34,133 | DEBUG   | train_full |   ── Tick  8000 ── airborne: 1/2 | RL decisions: 4 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:35,156 | DEBUG   | train_full |   ── Tick  9000 ── airborne: 1/2 | RL decisions: 4 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:35,758 | DEBUG   | train_full |   ── Tick 10000 ── airborne: 1/2 | RL decisions: 4 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:36,272 | DEBUG   | train_full |   ── Tick 11000 ── airborne: 1/2 | RL decisions: 4 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:36,688 | DEBUG   | train_full |   ── Tick 12000 ── airborne: 1/2 | RL decisions: 4 | reward: +2.00 | targets attacked: 1/5
2026-05-03 17:18:37,072 | DEBUG   | train_full |   Tick 12891 RTB:     agent be31019b.. landed
2026-05-03 17:18:37,076 | DEBUG   | train_full |   All agents returned to base at tick 12891 � ending episode
2026-05-03 17:18:37,077 | DEBUG   | train_full |   Episode utility: achieved=80 / oracle=240 (ratio=0.33) → ep_reward=+1.67
2026-05-03 17:18:37,106 | DEBUG   | train_full |   PPO update: policy_loss=-0.0032, value_loss=2.5261, entropy=1.2670, clip_frac=0.000
2026-05-03 17:18:37,122 | DEBUG   | train_full |   Recording exported: ep005_rl
2026-05-03 17:18:37,129 | INFO    | train_full | ep0005 [VAL]  ag=2 tg=5[3e+2s]  L1:e=2/3+0iso s=0/0+0iso  L2:clean         split=3/5  ou=160/240
2026-05-03 17:18:37,129 | INFO    | train_full | ep0005 [VAL]  RL=4d[A2 R2 N0] m=2/4  hit=1/5  RTB=Y  fd=1/1  t=12892  r= +3.67  u= 33%
2026-05-03 17:18:37,137 | DEBUG   | match_aou.rl.training.ppo_trainer | Saved PPO checkpoint: training_output\models\checkpoint_ep5.pt
2026-05-03 17:18:37,137 | INFO    | train_full | ========== Progress @ ep0005 | checkpoint saved | rolling 1ep ==========
2026-05-03 17:18:37,137 | INFO    | train_full |   Reward   :  +3.67  Δ +2.42 ↑    Utility :  33.3%  Δ  +8.3% ↑  Accuracy:  50.0%  Δ +16.7% ↑
2026-05-03 17:18:37,138 | INFO    | train_full |   Ticks/ep :  12892  Δ  -1490 ↓     Actions : A:50.0% R:50.0% N: 0.0%   Decisions: 4.00/ep
2026-05-03 17:18:37,138 | INFO    | train_full |   PPO loss : π=-0.0032  V=  2.53  H=1.267    Flags(window): (none)
2026-05-03 17:18:37,138 | INFO    | train_full | ========================================================================
2026-05-03 17:18:37,138 | INFO    | train_full |
======================================================================
2026-05-03 17:18:37,138 | INFO    | train_full | Training Complete!
2026-05-03 17:18:37,139 | INFO    | train_full | ======================================================================
2026-05-03 17:18:37,142 | DEBUG   | match_aou.rl.training.ppo_trainer | Saved PPO checkpoint: training_output\models\final_model.pt
2026-05-03 17:18:37,149 | INFO    | train_full | Total episodes:      5
2026-05-03 17:18:37,150 | INFO    | train_full | Total PPO updates:   5
2026-05-03 17:18:37,150 | INFO    | train_full | Avg policy loss:     -0.0055
2026-05-03 17:18:37,150 | INFO    | train_full | Avg value loss:      2.5751
2026-05-03 17:18:37,150 | INFO    | train_full | Avg reward (last 10): 1.90
2026-05-03 17:18:37,150 | INFO    | train_full | Avg accuracy (last 10): 46.7%
2026-05-03 17:18:37,150 | INFO    | train_full | Avg utility ratio (last 10): 30.0%
2026-05-03 17:18:37,151 | INFO    | train_full | Run summary written to: training_output\logs\run_summary.txt
2026-05-03 17:18:37,161 | INFO    | train_full | Highlights written to:  training_output\logs\highlights.txt
2026-05-03 17:18:37,161 | INFO    | train_full |
Outputs saved to: C:\Users\Itama\PycharmProjects\Multi_Agent_Task_Allocation_and_Adaptation\training_output
2026-05-03 17:18:37,161 | INFO    | train_full |   Logs:       training_output\logs/
2026-05-03 17:18:37,161 | INFO    | train_full |   Recordings: training_output\recordings/
2026-05-03 17:18:37,161 | INFO    | train_full |   Models:     training_output\models/
2026-05-03 17:18:37,161 | INFO    | train_full |   Scenarios:  training_output\scenarios/
Recording exported to 'training_output\recordings/ep005_rl Recording 064510 - 102019.jsonl'
exit=0
```

---

## 4. Files created/modified during run

> Listing of files in `training_output/` whose mtime is newer than the run
> marker (`.run_marker`, touched at 16:44 immediately before kicking off
> the run). Only the project's output tree is shown — third-party temp
> files (Pyomo `.nl` files in `%TEMP%`) are not.

### `training_output/logs/`
```
-rw-r--r-- 1 Itama 197609   36069 May  3 17:17 episode_0001.log
-rw-r--r-- 1 Itama 197609   38055 May  3 17:17 episode_0002.log
-rw-r--r-- 1 Itama 197609   37616 May  3 17:17 episode_0003.log
-rw-r--r-- 1 Itama 197609 3944300 May  3 17:18 episode_0004.log
-rw-r--r-- 1 Itama 197609   40668 May  3 17:18 episode_0005.log
-rw-r--r-- 1 Itama 197609    2198 May  3 17:18 highlights.txt
-rw-r--r-- 1 Itama 197609    1102 May  3 17:18 run_summary.txt
-rw-r--r-- 1 Itama 197609 4113242 May  3 17:18 training.log
```

(episode_0004.log is ~100× bigger than the others because the !TIMEOUT
forced flag pushed the episode out to 14,400 ticks — the full per-tick
DEBUG trace fills it.)

### `training_output/models/`  (only files touched by this run; the
historical `checkpoint_ep100.pt` … `checkpoint_ep5000.pt` from previous
runs are also present in this directory but were NOT modified)
```
-rw-r--r-- 1 Itama 197609 165893 May  3 17:18 actor_critic_final.pt
-rw-r--r-- 1 Itama 197609 498509 May  3 17:17 checkpoint_ep1.pt
-rw-r--r-- 1 Itama 197609 498893 May  3 17:17 checkpoint_ep2.pt
-rw-r--r-- 1 Itama 197609 499277 May  3 17:17 checkpoint_ep3.pt
-rw-r--r-- 1 Itama 197609 499661 May  3 17:18 checkpoint_ep4.pt
-rw-r--r-- 1 Itama 197609 500045 May  3 17:18 checkpoint_ep5.pt
-rw-r--r-- 1 Itama 197609 499883 May  3 17:18 final_model.pt
```

### `training_output/recordings/`
```
-rw-r--r-- 1 Itama 197609  3237721 May  3 17:17 ep001_rl Recording 064510 - 081341.jsonl
-rw-r--r-- 1 Itama 197609  3340985 May  3 17:17 ep001_validation Recording 064510 - 081649.jsonl
-rw-r--r-- 1 Itama 197609  4676180 May  3 17:17 ep002_rl Recording 064510 - 084307.jsonl
-rw-r--r-- 1 Itama 197609  4512205 May  3 17:17 ep002_validation Recording 064510 - 085822.jsonl
-rw-r--r-- 1 Itama 197609  3720215 May  3 17:17 ep003_rl Recording 064510 - 082953.jsonl
-rw-r--r-- 1 Itama 197609  4828631 May  3 17:17 ep003_validation Recording 064510 - 082954.jsonl
-rw-r--r-- 1 Itama 197609 11165391 May  3 17:18 ep004_rl Recording 064510 - 104510.jsonl
-rw-r--r-- 1 Itama 197609 11381563 May  3 17:18 ep004_validation Recording 064510 - 104510.jsonl
-rw-r--r-- 1 Itama 197609  8254954 May  3 17:18 ep005_rl Recording 064510 - 102019.jsonl
-rw-r--r-- 1 Itama 197609  1687142 May  3 17:18 ep005_validation Recording 064510 - 072747.jsonl
```

### `training_output/scenarios/`
```
-rw-r--r-- 1 Itama 197609 10013 May  3 17:16 episode_0000_scenario.json
-rw-r--r-- 1 Itama 197609 10402 May  3 17:17 episode_0001_scenario.json
-rw-r--r-- 1 Itama 197609 13177 May  3 17:17 episode_0002_scenario.json
-rw-r--r-- 1 Itama 197609 13563 May  3 17:17 episode_0003_scenario.json
-rw-r--r-- 1 Itama 197609 10746 May  3 17:18 episode_0004_scenario.json
```

(Scenario files are written as 0-indexed `episode_NNNN_scenario.json` —
the file name's `NNNN` is `episode_index`, while episode log/recording
names use 1-based `epNNNN` / `epNNN`.)

---

## 5. Conditional logs not observed

> For each: file:line — message — one-line guess at trigger condition
> (based purely on the surrounding code).

### Dead code paths (modules not imported by `train_full.py`)

- [src/match_aou/rl/training/oracle.py:157](src/match_aou/rl/training/oracle.py:157) — `MATCH-AOU solver returned no solution`. Would only fire if `Oracle` were instantiated and `solve()` called; train_full inlines its own `solve_match_aou` instead.
- [src/match_aou/rl/training/oracle.py:163](src/match_aou/rl/training/oracle.py:163) — `Failed to solve MATCH-AOU: <e>`. Same — dead in active flow.
- [src/match_aou/rl/training/episode_initializer.py:68-176](src/match_aou/rl/training/episode_initializer.py:68) — `Initializing Episode` banner / `Solving MATCH-AOU...` / task-set summary / `Launched <N> aircraft` / `Waiting for takeoff...` / `Getting initial observations...` / `Episode initialized: ...` / `Skipping <agent_id>: no assignments` / `Launched <name>` / `Failed to launch <id>: <e>` / `Failed to get observation for <id>: <e>`. None fire — `EpisodeInitializer` is not imported by train_full.

### Run-only-on-demand paths

- [src/match_aou/utils/blade_utils/scenario_generator.py:1343-1386](src/match_aou/utils/blade_utils/scenario_generator.py:1343) — `Aircraft pool: …`, `Facility pool: …`, `Episode <N>: <path>`, `Aircraft (<C>): …`, `Facilities (<C>): …`, `RED airbases: <N>`, `Base: (<lat>, <lon>)`. Only fire when running `python -m match_aou.utils.blade_utils.scenario_generator <scenario.json>` (the manual scenario-inspection command in CLAUDE.md §5).
- [src/match_aou/solvers/match_aou_MINLP_solver.py:223-235](src/match_aou/solvers/match_aou_MINLP_solver.py:223) — `No solution found …`, `Assigned Tasks:`, `Agent <id> assigned to steps:`, `  Task <i>, Step <j>`, `Unassigned Tasks:`, `Task <j> is unassigned (Utility: <U>)`. `display_solution()` is defined but never called from `train_full.py`.
- [src/match_aou/rl/training/ppo_trainer.py:370](src/match_aou/rl/training/ppo_trainer.py:370) — `Loaded PPO checkpoint: <path> (episode <N>)`. Only fires when `load_checkpoint()` is called; not invoked in the standard training flow (trainer always starts fresh).

### Healthy-run paths (only fire on degenerate / pathological scenarios)

- [train_full.py:194](train_full.py:194) — `Scenario duration … > max_steps …`. Only fires when scenario JSON's `duration` exceeds `--max-ticks`.
- [train_full.py:248](train_full.py:248) — `No tasks or agents to solve`. Empty-input guard in `solve_match_aou`.
- [train_full.py:255](train_full.py:255) — `MATCH-AOU returned empty solution`. Solver returned no allocation.
- [train_full.py:362](train_full.py:362) — `Task split: … (chain check skipped: no radar range)`. Fleet has no radar range data.
- [train_full.py:392](train_full.py:392) / [397](train_full.py:397) — `Discovery chain (split): isolated=<I> exceeds partial budget …`. More isolated targets than the partial budget can absorb.
- [train_full.py:445](train_full.py:445) / [450](train_full.py:450) — `Discovery chain (split): no valid split after <N> attempts …`. 20-retry rejection sampling exhausted.
- [train_full.py:655](train_full.py:655) — `Tick … ACTION: <action>`. Fires for any BLADE action string that doesn't match the four `_RE_*` regexes (`handle_aircraft_attack`, `move_aircraft`, `launch_aircraft_from_airbase`, `return_to_base`).
- [train_full.py:722](train_full.py:722) / [727](train_full.py:727) / [759](train_full.py:759) — `Validation: no agents found, skipping` / `Validation: no tasks found, skipping` / `Validation: solver returned empty solution, skipping`. Degenerate validation scenarios.
- [train_full.py:873](train_full.py:873) / [888](train_full.py:888) — `Tick … [VAL END-ZONE] …` and per-aircraft dump. Only fires in the last 100 ticks before `--max-ticks`; validation episodes in this run all terminated well before.
- [train_full.py:898](train_full.py:898) — `Validation ended at tick … terminated/truncated`. Validation hit env terminate/truncate before all agents returned.
- [train_full.py:947](train_full.py:947) — `Dropped reachable targets (oracle chose not to plan): …`. Oracle plan didn't include some reachable targets (audit-only).
- [train_full.py:952](train_full.py:952) — `ANOMALY: unreachable target(s) attacked: …`. Should never happen in a healthy run; would indicate a budget/reachability bug.
- [train_full.py:958](train_full.py:958) — `Oracle plan incomplete in execution — missed: …`. Validation play-through failed to complete every step in the oracle plan.
- [train_full.py:969](train_full.py:969) — `Failed to export validation recording: <e>`. BLADE `export_recording` raised.
- [train_full.py:1050](train_full.py:1050) / [1055](train_full.py:1055) — `No attacking agents found!` / `No tasks generated!`. Misconfigured scenario.
- [train_full.py:1200](train_full.py:1200) — `Partial solution empty, skipping episode`. Skips the episode entirely.
- [train_full.py:1334](train_full.py:1334) — `Scenario has <N> agents but MAX_AGENTS=5`. Scenario produced more agents than the critic was sized for.
- [train_full.py:1351](train_full.py:1351) — `Tick <T>: Executor error (skipping): <e>`. `BladeExecutorMinimal.next_action` raised mid-loop.
- [train_full.py:1400](train_full.py:1400) — `Tick <T>: Can't observe <agent_id>: <e>`. `build_observation_vector` raised on a tick.
- [train_full.py:1518](train_full.py:1518) — `RL action <A> invalid for <agent_id>: <e>`. `plan_edit_to_blade_action` raised (RL picked an attack on a slot whose target is gone).
- [train_full.py:1564](train_full.py:1564) / [1579](train_full.py:1579) — RL END-ZONE block + per-aircraft dump. Only fires in the last 100 ticks before `--max-ticks`. The single RL episode that did hit the cap (ep4 with !TIMEOUT) IS observed in this run, but its END-ZONE lines fall in the elided middle of the runtime sample.
- [train_full.py:1592](train_full.py:1592) — `Episode ended at tick … terminated/truncated …`. Episode terminated by env (not by all agents RTB).
- [train_full.py:1651](train_full.py:1651) / [src/match_aou/rl/training/ppo_trainer.py:244](src/match_aou/rl/training/ppo_trainer.py:244) — `No transitions collected, skipping PPO update` / `Buffer empty, skipping update`. Episode produced zero RL decisions (no triggers fired).
- [train_full.py:1668](train_full.py:1668) — `Failed to export recording: <e>`. BLADE `export_recording` raised on RL episode.
- [train_full.py:2530](train_full.py:2530) — `Time-feasibility cap: not computed (empty pool)`. Fires only when no eligible aircraft pool exists after `--allowed-aircraft` filtering.
- [train_full.py:2532](train_full.py:2532) — `Time-feasibility cap: <K> km one-way [manual override via --time-feasible-max-km]`. Only fires when the user passes `--time-feasible-max-km`.
- [train_full.py:2703](train_full.py:2703) / [2706](train_full.py:2706) — `!CRASH ep<NNNN> …` traceback + follow-up. Fires when `train_episode` raises any exception.
- [train_full.py:2753](train_full.py:2753) — `→ Replaying flagged episode for recording: <name>`. Only fires when `should_record` was False (i.e. cadence didn't already cover the flagged episode). In this capture `--record-every 1` made every episode recordable, so the replay path was never taken even though three episodes carried flags.
- [train_full.py:2771](train_full.py:2771) — `Flagged-episode replay failed: <e>`. Replay raised; same pre-condition as above.
- [train_full.py:2806](train_full.py:2806) — `=== Checkpoint saved (epNNNN) | rolling avg reward (last <N>): <R> ===`. Only fires when `--progress-every 0`; this run used `--progress-every 1` so the multi-line progress block fired instead.
- [train_full.py:2841](train_full.py:2841) / [2856](train_full.py:2856) — `Failed to write run_summary.txt` / `Failed to write highlights.txt`. End-of-run text writes raised.
- [src/match_aou/utils/blade_utils/scenario_generator.py:678](src/match_aou/utils/blade_utils/scenario_generator.py:678) — `Target '<name>' is unreachable by all agents - expected behavior for stretch targets`. Reachability audit flagged a target as unreachable.
- [src/match_aou/utils/blade_utils/scenario_generator.py:776](src/match_aou/utils/blade_utils/scenario_generator.py:776) — `Target placement: <E> easy (≤<X>km), <S> stretch (<MIN>–<MAX>km)`. Fires only when stretch zone is in use; ep1's run had the stretch zone collapsed by the time-feasibility cap, so this didn't fire on ep1 but DOES fire on later episodes (in the elided middle of the runtime sample).
- [src/match_aou/utils/blade_utils/scenario_generator.py:784](src/match_aou/utils/blade_utils/scenario_generator.py:784) — `Stretch targets disabled: fleet range gap (<G>km) too small for differentiation`. Fleet has near-identical ranges (gap ≤ 50 km).
- [src/match_aou/utils/blade_utils/scenario_generator.py:834](src/match_aou/utils/blade_utils/scenario_generator.py:834) — `Stretch target fell back to easy zone`. Stretch placement attempts all failed.
- [src/match_aou/utils/blade_utils/scenario_generator.py:1095](src/match_aou/utils/blade_utils/scenario_generator.py:1095) — `Discovery chain: could not connect target '<name>' within zone bounds [<MIN>-<MAX> km]; leaving isolated`. Generation-time discovery chain couldn't relocate a target into in-zone radar reach.
- [src/match_aou/utils/blade_utils/scenario_generator.py:1298](src/match_aou/utils/blade_utils/scenario_generator.py:1298) / [1304](src/match_aou/utils/blade_utils/scenario_generator.py:1304) — `No fuel tier for class '<cls>'; keeping template fuel` / `Cannot compute fuel for '<cls>' (speed/fuelRate invalid); keeping template fuel`. Aircraft class isn't in `CLASS_RANGE_TIERS`, or its speed/fuelRate is invalid.
- [src/match_aou/solvers/match_aou_MINLP_solver.py:197](src/match_aou/solvers/match_aou_MINLP_solver.py:197) — `Model not solved to acceptable optimality. Check constraints and inputs.` Bonmin returned a non-optimal termination condition.
- (vendored BLADE) `Game.py:799` — `<exception repr>` printed by BLADE `Game` on internal exception.
