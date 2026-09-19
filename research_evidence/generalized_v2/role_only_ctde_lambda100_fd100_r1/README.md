# Role-only acting-ego CTDE, `gae_lambda = 1.0`, FD100 development diagnostic — compact evidence package

**Development evidence only: one seed stream, repeated evaluation on the same frozen
development worlds, not randomized, not confirmatory.**

## Formal status

- Measured code SHA `68055e39768d5fa601e5960a9f08823b9e65c08f`, the role-only acting-ego
  implementation the primary comparator was measured on. The run executed from an **isolated
  detached worktree** (`C:\grolelambda1`, no branch), **not** from PR #75's later
  explicit-readout head, under the user-authorized bounded plan in `originals/authorized_plan.json`.
- Intervention: **`ctde.gae_lambda 0.95 → 1.0`** plus a fresh `output_dir`; nothing else in the
  resolved `train_config`.
- The run **has completed**.
- **Scientific measurement verdict: PENDING GPT review.** Nothing here is an interpretation or a
  verdict.
- **Primary comparison:** same-code role-only `λ = 0.95` versus `λ = 1.0`. The symmetric
  (`6ed964a…`) and explicit-readout (`1a1e0c9…`) arms are **secondary cross-version context
  only**, compared through update 100.
- Bitwise identity between the two λ arms is not claimed (BLADE run-to-run timing
  nondeterminism; separate executions).
- The confirmatory profile is untouched. **No new run is authorized** and no merge is authorized.

## Arms

| Key | Run ID (original directory under `C:\gruns\`, authoritative, external) | Measured code SHA | Critic | `gae_lambda` |
|---|---|---|---|---|
| `lambda100` (new) | `graph_rl_v2_role_only_ctde_lambda100_fd100_r1_seed3000000_68055e3` | `68055e39768d5fa601e5960a9f08823b9e65c08f` | role-only acting-ego central state, mean-pool readout | 1.0 |
| `role_only_lambda095` (**primary**) | `graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` | `68055e39768d5fa601e5960a9f08823b9e65c08f` | same | 0.95 |
| `symmetric` (secondary, cross-version) | `graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` | `6ed964a1abd09de2130aee3d0d314c8f32165056` | symmetric central state; 150 updates | 0.95 |
| `explicit_readout` (secondary, cross-version) | `graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9` | `1a1e0c953c54e9d3f46c871158d5ab6bdd881f24` | role-only state + `[mean pool ; acting-ego embedding]` readout | 0.95 |

## Identity and configuration

- **Checkout:** the preflight (16 / 16 checks passed, all written before the first episode)
  records HEAD `68055e3…`, detached, `git status --porcelain --untracked-files=all` empty, the
  role-only acting-ego code present and the explicit readout absent. `run_config.json` provenance:
  commit `68055e3…`, `dirty = false`, `repo_root C:\grolelambda1`, `branch HEAD`, `match_aou`
  imported from the worktree.
- **BLADE:** imported from the main checkout's editable install; the preflight verified that the
  vendored engine tree at the measured SHA equals the loaded checkout's tree, with no local
  modification.
- **Configuration route:** `originals/train_config_preset.json` is the comparator's
  `run_config.json:/train_config` with only `/ctde/gae_lambda` and `/output_dir` replaced,
  resolved by the trainer's own `--config` loader. The invocation is
  `python -m match_aou.rl.training.graph_train --config <run_dir>\train_config_preset.json`
  under `conda run -n nlp_env --no-capture-output`, `PYTHONPATH=src`, cwd `C:\grolelambda1`
  (`originals/launch.cmd`).
- **Resolved `train_config` versus the primary comparator differs in exactly
  `/ctde/gae_lambda` (0.95 → 1.0) and `/output_dir`** (flattened JSON-path comparison; no
  unauthorized difference). Versus explicit readout: the same two paths; versus symmetric:
  additionally `/n_iterations`.
- **`config_source.resolved_from` differs** — `config_file` (the preset, no CLI overrides) for
  the new run, `cli_defaults` for the comparator. This is **provenance** of how an identical
  `train_config` was resolved, **not** a scientific configuration difference. Every other
  differing `run_config.json` path is provenance (git `repo_root` / `branch`, invocation argv /
  cwd, `collected_at`, `match_aou` path) or the mirrored `/training/ctde/gae_lambda`; the list is
  in `evidence_summary.json:/config_boundaries`. The seed schedule block equals the comparator's.
- **Benchmark:** manifest `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`,
  file SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`, `development`
  profile only (20 worlds, 60 members), held out against `[3000000, 3001200)` over the entire
  manifest, overlap 0 (all arms).
- **Comparator originals unchanged:** role-only λ0.95 bytes equal
  `../acting_ego_ctde_fd100_r1/artifact_manifest.json`; symmetric equal
  `../semantic_ctde_grad_diag_r1/artifact_manifest.json`; explicit readout equal
  `../explicit_ego_readout_ctde_fd100_r1/artifact_manifest.json`.

## Completion and accounting (`lambda100`)

100 / 100 productive updates; 800 / 800 successful training episodes over 800 attempts (budget
1200), 0 failed, 0 replacements; 2981 transitions = `train_records` Σ `n_transitions` = credit
rows; `accounting_reconciled = true`; FD100: 0 clean, 800 damaged (421 MILD, 379 SEVERE), 800
events, 800 FD wakes; 5 evaluation rounds, 300 / 300 successful; 100 gradient rows (99 defined);
`episode_failures.jsonl` empty; `native_exit_code.txt` `0`; no `Traceback` / `CRASH` in the
console. Wall clock 17:56:51 → 18:40:16 local, 2026-09-19.

## Held-out behaviour (common range)

Macro `SEVERE − MILD P(SELF_PRESERVATION_ABORT)` at the certified ego's immediate-FD wake. Every
round of every arm: 10 / 10 base cells defined, 20 / 20 matched groups metric-eligible, 60 / 60
members, directional switches **0 / 20**, reverse switches 0.

| Update | **Role-only λ0.95 (primary)** | **Role-only λ1.0 (new)** | Symmetric λ0.95 (cross-version) | Explicit λ0.95 (cross-version) |
|---|---|---|---|---|
| 0 | −0.0005158 | −0.0005158 | −0.0005158 | −0.0005158 |
| 25 | −0.0000357 | +0.0000138 | +0.0000907 | −0.0000200 |
| 50 | +0.0000326 | +0.0000046 | +0.0020784 | +0.0000009 |
| 75 | +0.0076414 | +0.0000168 | +0.0427981 | +0.0000005 |
| 100 | −0.0010906 | +0.0000014 | −0.0000086 | +0.0000004 |

## Structural λ = 1 identity (every CTDE transition, iterations 0–99)

| Max absolute error | λ1.0 (n = 2981; 2181 nonterminal, 800 terminal) | λ0.95 (n = 2982; 2182 / 800) |
|---|---|---|
| `raw_advantage − (episode_reward − value_old)` | 5.6e-17 (0 rows > 1e-9) | 0.349 (2182 rows > 1e-9) |
| `value_target − episode_reward` | 5.6e-17 (0 rows > 1e-9) | 0.349 (2182 rows > 1e-9) |
| nonterminal `td − (r + γ·V_next − V)` | 0 | 0 |
| terminal `td − (r − V)` | 0 | 0 |
| GAE `A − td − γλ·A_next` (nonterminal) | 5.6e-17 | 1.1e-16 |
| nonterminal `transition_reward` / terminal `transition_reward − R` | 0 / 0 | 0 / 0 |

With `γ = 1`, `λ = 1` and terminal-only reward the actor's total advantage telescopes to
`R − V_t` and the value target to `R`: **no intermediate `V_{t+k}` — including a value conditioned
on a different acting ego — survives in the λ = 1 actor advantage.** For λ0.95 the errors
measure the retained intermediate bootstrap and are reported, not checked.

## Critic and credit (immediate-FD rows of the FD-selected ego, 800 per arm)

Within-update `SEVERE − MILD` median (0–99, 99 updates / 50–99, 50 updates):

| Quantity | Role-only λ0.95 | Role-only λ1.0 | Symmetric | Explicit |
|---|---|---|---|---|
| `value_old` | −0.0205 / −0.0236 | −0.0390 / −0.0892 | −0.000028 / −0.0000019 | −0.00237 / −0.0125 |
| `value_target` | −0.330 / −0.369 | −0.367 / −0.421 | −0.317 / −0.361 | −0.327 / −0.372 |
| `raw_advantage` | −0.319 / −0.357 | −0.317 / −0.328 | −0.316 / −0.360 | −0.324 / −0.356 |
| `td_residual` | −0.00178 / −0.00306 | +0.00423 / +0.00104 | −0.000089 / −0.000109 | −0.00215 / −0.00125 |

`value_old` by window (0–24 / 25–49 / 50–74 / 75–99): λ0.95 −0.0024 / −0.0549 / −0.0132 / −0.0583;
λ1.0 −0.0085 / −0.0179 / −0.0713 / −0.1245.

**Action-credit association** — mean normalized advantage of sampled ABORT minus sampled PLAN,
`ABORT / PLAN` counts in brackets. **Descriptive sampled-action associations, not counterfactual
Q values.**

| Window | Severity | Role-only λ0.95 | Role-only λ1.0 | Symmetric | Explicit |
|---|---|---|---|---|---|
| 0–99 | MILD | −1.039 (110/311) | −1.059 (109/312) | −1.090 (120/301) | −1.089 (109/312) |
| 0–99 | SEVERE | +0.340 (103/275) | +0.350 (99/279) | +0.369 (109/269) | +0.394 (99/278) |
| 50–74 | MILD | −1.399 (18/89) | −1.465 (18/89) | −1.356 (22/85) | −1.335 (19/88) |
| 50–74 | SEVERE | +0.675 (16/76) | +0.902 (14/78) | +0.691 (19/73) | +0.861 (16/75) |
| 75–99 | MILD | −0.947 (11/96) | −0.674 (13/94) | −1.021 (21/86) | −1.028 (12/95) |
| 75–99 | SEVERE | +0.375 (13/80) | +0.074 (15/78) | +0.232 (21/72) | +0.299 (14/79) |

## Local diagnostic and owner transitions (nonterminal immediate-FD rows)

**Under λ = 1 the one-step TD residual remains a diagnostic of consecutive values, but it is NOT
the actor's total advantage: total actor credit telescopes to `R − V_t`**, and
`future = A − td = A_{t+1} = R − V_{t+1}`. With terminal-only reward and `γ = 1` a nonterminal
`td_residual` equals `ΔV = V_{t+1} − V_t` exactly (verified).

| Arm (nonterminal n) | median `|td|/|A|` | median `|td|` | median `|future|` | sign(td) ≠ sign(A) | `|future| > |local|` |
|---|---|---|---|---|---|
| Role-only λ0.95 (706) | 0.0470 | 0.0068 | 0.188 | 0.424 | 0.921 |
| Role-only λ1.0 (705) | 0.0398 | 0.0072 | 0.206 | 0.479 | 0.919 |
| Symmetric (699) | 0.0031 | 0.00052 | 0.198 | 0.509 | 0.980 |
| Explicit (708) | 0.0125 | 0.0022 | 0.193 | 0.492 | 0.959 |

| Split | Arm | n (fraction) | median / mean `|td|` | median `|td|/|A|` | median / mean ΔV | pstdev / IQR ΔV | sign(td) ≠ sign(A) |
|---|---|---|---|---|---|---|---|
| same ego next | λ0.95 | 377 (0.534) | 0.0050 / 0.0196 | 0.0275 | +0.00014 / −0.00023 | 0.0463 / 0.0098 | 0.387 |
| same ego next | λ1.0 | 380 (0.539) | 0.0053 / 0.0224 | 0.0238 | −0.00021 / +0.0018 | 0.0476 / 0.0104 | 0.497 |
| different ego next | λ0.95 | 329 (0.466) | 0.0131 / 0.0403 | 0.0862 | +0.0048 / +0.0233 | 0.0675 / 0.0395 | 0.465 |
| different ego next | λ1.0 | 325 (0.461) | 0.0117 / 0.0442 | 0.0785 | +0.0046 / +0.0322 | 0.0748 / 0.0498 | 0.458 |

Next-wake categories (only observed cells): every different-ego next decision is `ordinary`
(λ0.95 329, λ1.0 325); same-ego next decisions are `ordinary` / `post_fd_boundary` (λ0.95 183 /
194, λ1.0 184 / 196). Symmetric and explicit-readout splits, per-cell statistics, severity counts,
iteration-window splits and within-update `SEVERE − MILD` medians of `V_t`, `V_next`, `ΔV`, `td`
and `future` are in `evidence_summary.json:/arms/<arm>/owner_transition_audit`.

## Actor gradient (99 defined updates per arm; windows of 25 defined updates)

| | Role-only λ0.95 | Role-only λ1.0 | Symmetric | Explicit |
|---|---|---|---|---|
| 50–74 FD+ / non-FD+ / total+ / FD+∧total− | 15 / 15 / 15 / 2 | 14 / 11 / 12 / 4 | 20 / 21 / 18 / 3 | 12 / 7 / 10 / 4 |
| 50–74 FD median / mean | +0.00213 / −0.00545 | +0.0000472 / +0.00742 | +0.0195 / +0.00683 | −0.00128 / −0.00163 |
| 50–74 non-FD median / total median | +0.00298 / +0.00581 | −0.000333 / −0.0000603 | +0.0190 / +0.0533 | −0.00139 / −0.00229 |
| 75–99 FD+ / non-FD+ / total+ / FD+∧total− | 18 / 11 / 13 / 6 | 18 / 10 / 12 / 7 | 13 / 9 / 12 / 2 | 15 / 14 / 13 / 3 |
| 75–99 FD median / mean | +0.00122 / −0.000228 | +0.000525 / +0.000802 | +0.000257 / −0.0128 | +0.00198 / +0.00310 |
| 75–99 non-FD median / total median | −0.00123 / +0.000509 | −0.000387 / −0.000137 | −0.000663 / −0.000679 | +0.000171 / +0.000418 |
| whole run FD+ / FD+∧non-FD− / FD+∧total− | 63 / 29 / 15 | 64 / 35 / 19 | 64 / 22 / 11 | 57 / 27 / 12 |
| entropy sign flips / min cosine | 1 / 0.878 | 0 / 0.995 | 0 / 0.988 | 0 / 0.892 |
| max reconstruction relative error | 5.9e-7 | 6.4e-7 | 4.7e-7 | 5.4e-7 |

## Files

| Path | Content |
|---|---|
| `evidence_summary.json` | formal status, SHAs, config boundaries (incl. the `config_source` provenance note), per-arm identity / validity / accounting / benchmark, trajectories, structural identities (λ arms), credit, action-credit, gradient and owner-transition blocks, new-run specifics (preflight checks, launch, console summary), integrity checks, definitions |
| `artifact_manifest.json` | SHA-256 and bytes of every top-level file of all four run directories (plus the new run's `train_config_preset.json`); tree digests of `checkpoints/`, `plots/`, `scenarios/` (per file for checkpoints) |
| `originals/` | byte-identical copies of the new run's `authorized_plan.json`, `preflight.json`, `train_config_preset.json`, `launch.cmd`, `launch_record.json`, `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `train_actor_gradient_diagnostics.jsonl`, `episode_failures.jsonl`, `invocation_start_local.txt`, `native_exit_code.txt` |
| `derived/<λ arm>/all_transition_rows.jsonl` | EVERY CTDE transition with its next decision's value and advantage (structural identities, TD / GAE recurrence) |
| `derived/<arm>/immediate_fd_credit_rows.jsonl` | immediate-FD FD-selected-ego credit rows with the episode-terminal flag |
| `derived/<arm>/owner_transition_rows.jsonl` | nonterminal rows joined to the next global decision |
| `derived/<arm>/actor_gradient_rows.jsonl` | per-update gradient projection |
| `derived/<arm>/evaluation_trajectory.jsonl` | evaluation rounds 0–100 |
| `scripts/extract_evidence.py` | standard-library extractor and checker |

Not committed (identified by SHA-256 in `artifact_manifest.json`): each run's
`train_credit_diagnostics.jsonl`, `episode_outcomes.jsonl`, console log, checkpoints, plots and
scenarios.

## Reproduce and verify

```bash
python research_evidence/generalized_v2/role_only_ctde_lambda100_fd100_r1/scripts/extract_evidence.py --check
python research_evidence/generalized_v2/role_only_ctde_lambda100_fd100_r1/scripts/extract_evidence.py --from-derived
```

`--check` reads all four original run directories, requires byte identity of every generated
file and runs 138 integrity checks. `--from-derived` needs no run directory: it recomputes every
mechanism block from `derived/` and re-runs the row-level identity checks (λ = 1 telescoping,
TD / GAE recurrence, terminal-only reward, owner-transition joins). All quantities are descriptive
and non-counterfactual.
