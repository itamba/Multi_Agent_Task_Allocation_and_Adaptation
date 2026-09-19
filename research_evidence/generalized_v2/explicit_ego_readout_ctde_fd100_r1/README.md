# Explicit acting-ego readout CTDE FD100 development diagnostic — compact evidence package

**Development evidence only; not randomized; not confirmatory.**

## Formal status

- Implementation candidate `1a1e0c953c54e9d3f46c871158d5ab6bdd881f24` (PR #75) was
  **GPT-approved for measurement**; the run executed under the user-authorized bounded plan in
  `originals/authorized_plan.json`.
- The run **has completed**.
- **Scientific measurement verdict: PENDING GPT review.** Nothing here is an interpretation or a
  verdict.
- The **primary comparator is the role-only acting-ego run** (`68055e3…`); the intended
  code-level difference is the critic readout. The **symmetric FD100 run** (`6ed964a…`) is
  **secondary context**, compared only through update 100.
- The confirmatory profile is untouched. **No new run is authorized** and no merge is authorized.

## Arms

| Key | Run ID (original directory under `C:\gruns\`, authoritative, external) | Measured code SHA | Critic |
|---|---|---|---|
| `explicit_readout` | `graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9` | `1a1e0c953c54e9d3f46c871158d5ab6bdd881f24` | role-only acting-ego central state + readout `[mean pool ; acting-ego embedding]` (`mean_pool_plus_acting_ego_v1`, value head input 128) |
| `role_only` (primary) | `graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` | `68055e39768d5fa601e5960a9f08823b9e65c08f` | role-only acting-ego central state, mean-pool readout |
| `symmetric` (secondary) | `graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` | `6ed964a1abd09de2130aee3d0d314c8f32165056` | symmetric central state, mean-pool readout; 150 updates |

- **Configuration boundaries** (resolved `train_config`): explicit vs role-only differ only in
  `/output_dir`; explicit vs symmetric differ only in `/n_iterations` and `/output_dir`. The
  critic differences are **code-level** and are stated per arm, not hidden as configuration
  equality.
- **Provenance:** `run_config.json:/provenance/git` records commit `1a1e0c9…`, `dirty = false`,
  0 dirty paths; the preflight records checkout head, origin branch head and PR #75 head equal
  to it, with an empty `git status --porcelain --untracked-files=all`, and the readout id and
  value-head shape read from that code.
- **Benchmark:** manifest `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`,
  file SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`, `development`
  profile only, held out against `[3000000, 3001200)` over the whole manifest, overlap 0 (all
  three arms).
- **Comparator originals unchanged:** the role-only run's bytes equal
  `../acting_ego_ctde_fd100_r1/artifact_manifest.json`; the symmetric run's equal
  `../semantic_ctde_grad_diag_r1/artifact_manifest.json`.

## Completion and accounting (explicit readout)

100 / 100 productive updates; 800 / 800 successful training episodes over 800 attempts (budget
1200), 0 failed, 0 replacements; 3003 transitions = `train_records` Σ `n_transitions` = credit
rows; `accounting_reconciled = true`; FD100: 0 clean, 800 damaged (421 MILD, 379 SEVERE), 800
events, 800 FD wakes; 5 evaluation rounds, 300 / 300 successful; 100 gradient rows (99 defined);
`episode_failures.jsonl` empty; `native_exit_code.txt` `0`; no Traceback. Wall clock 16:24:52 →
17:01:27 local, 2026-09-19.

## Reproduced metrics (common range: iterations 0–99, evaluation through update 100)

Held-out macro `SEVERE − MILD P(SELF_PRESERVATION_ABORT)`; every round of every arm: 10 / 10 base
cells, 20 / 20 matched groups eligible, 60 / 60 members, directional switches **0 / 20**, reverse 0.

| Update | Symmetric | Role-only | Explicit readout |
|---|---|---|---|
| 0 | −0.000516 | −0.000516 | −0.000516 |
| 25 | +0.0000907 | −0.0000357 | −0.0000200 |
| 50 | +0.00208 | +0.0000326 | +0.00000094 |
| 75 | +0.0428 | +0.00764 | +0.00000052 |
| 100 | −0.0000086 | −0.00109 | +0.00000043 |

Immediate-FD rows of the FD-selected ego (800 per arm); within-update `SEVERE − MILD` median
(0–99, 99 updates / 50–99, 50 updates):

| Quantity | Symmetric | Role-only | Explicit readout |
|---|---|---|---|
| `value_old` | −0.0000279 / −0.0000019 | −0.0205 / −0.0236 | −0.00237 / −0.0125 |
| `value_target` | −0.317 / −0.361 | −0.330 / −0.369 | −0.327 / −0.372 |
| `raw_advantage` | −0.316 / −0.360 | −0.319 / −0.357 | −0.324 / −0.356 |
| `td_residual` | −0.0000886 / −0.000109 | −0.00178 / −0.00306 | −0.00215 / −0.00125 |

`value_old` by window (0–24 / 25–49 / 50–74 / 75–99): symmetric −0.00062 / +0.00011 / −0.00003 /
+0.000007; role-only −0.0024 / −0.0549 / −0.0132 / −0.0583; explicit +0.0050 / −0.0025 / −0.0361 /
−0.0016.

| Nonterminal immediate-FD credit locality | Symmetric (699) | Role-only (706) | Explicit (708) |
|---|---|---|---|
| median `|td|/|A|` | 0.00307 | 0.0470 | 0.0125 |
| median `|local|` / median `|future|` | 0.00052 / 0.198 | 0.0068 / 0.188 | 0.0022 / 0.193 |
| mean `|local|` / mean `|future|` | 0.0547 | 0.123 | 0.0624 |
| fraction `|future| > |local|` | 0.980 | 0.921 | 0.959 |

| Gradient (25 / 25 defined each window) | Symmetric | Role-only | Explicit |
|---|---|---|---|
| 50–74 FD+ / non-FD+ / total+ / FD+∧total− | 20 / 21 / 18 / 3 | 15 / 15 / 15 / 2 | 12 / 7 / 10 / 4 |
| 50–74 FD median / mean | +0.0195 / +0.00683 | +0.00213 / −0.00545 | −0.00128 / −0.00163 |
| 75–99 FD+ / non-FD+ / total+ / FD+∧total− | 13 / 9 / 12 / 2 | 18 / 11 / 13 / 6 | 15 / 14 / 13 / 3 |
| 75–99 FD median / mean | +0.00026 / −0.0128 | +0.00122 / −0.00023 | +0.00198 / +0.00310 |
| whole run FD+ / FD+∧non-FD− / FD+∧total− (of 99) | 64 / 22 / 11 | 63 / 29 / 15 | 57 / 27 / 12 |
| entropy sign flips (of 99) / min cosine | 0 / 0.988 | 1 / 0.878 | 0 / 0.892 |

## Owner-transition credit audit (read-only)

For every NONTERMINAL immediate-FD row of the FD-selected ego (iterations 0–99), the next
decision is the row with `episode_decision_ordinal + 1` in the same episode's global decision
sequence. Verified in every arm: ordinals contiguous with no duplicate or gap, one seed per
episode, next ordinal = current + 1, `td_residual = r + γ·V_next − V_t` (max error 0) and
`future = γλ·A_{t+1}` (max error ≤ 1.1e-16), and the joined count equals the nonterminal
denominator. **Structure:** every nonterminal row has `transition_reward = 0` and `γ = 1`, so
here `td_residual` is exactly `delta_value = V_{t+1} − V_t`.

| Split (fraction of nonterminal) | Arm | n | median / mean `|td|` | median `|td|/|A|` | median / mean ΔV | pstdev / IQR ΔV | sign(td) ≠ sign(A) | `|future|>|local|` |
|---|---|---|---|---|---|---|---|---|
| same ego next | symmetric | 368 (0.527) | 0.00053 / 0.0168 | 0.00228 | −8.5e-7 / −0.0036 | 0.0744 / 0.0011 | 0.511 | 0.984 |
| same ego next | role-only | 377 (0.534) | 0.00505 / 0.0196 | 0.0275 | +0.00014 / −0.00023 | 0.0463 / 0.0098 | 0.387 | 0.944 |
| same ego next | explicit | 386 (0.545) | 0.00118 / 0.0107 | 0.00628 | −3.5e-6 / +0.0042 | 0.0377 / 0.0028 | 0.492 | 0.985 |
| different ego next | symmetric | 331 (0.474) | 0.00051 / 0.0123 | 0.00418 | −5.2e-6 / +0.00092 | 0.0414 / 0.00097 | 0.508 | 0.976 |
| different ego next | role-only | 329 (0.466) | 0.0131 / 0.0403 | 0.0862 | +0.0048 / +0.0233 | 0.0675 / 0.0395 | 0.465 | 0.894 |
| different ego next | explicit | 322 (0.455) | 0.0039 / 0.0242 | 0.0310 | +0.0016 / +0.0146 | 0.0536 / 0.0138 | 0.491 | 0.929 |

Next-wake categories (only observed cells): every different-ego next decision is `ordinary`;
same-ego next decisions are `ordinary` or `post_fd_boundary`. Per-cell statistics, severity
counts per split and within-update `SEVERE − MILD` medians of `V_t`, `V_next`, `delta_value`,
`td_residual` and `future_component` (overall and per split, with denominators) are in
`evidence_summary.json:/arms/<arm>/owner_transition_audit`.

## Files

| Path | Content |
|---|---|
| `evidence_summary.json` | formal status, SHAs, config boundaries, per-arm identity / validity / accounting / benchmark, trajectories, credit, gradient and owner-transition blocks, integrity checks, definitions |
| `artifact_manifest.json` | SHA-256 and bytes of every top-level file of all three run directories; tree digests of `checkpoints/`, `plots/`, `scenarios/` (per file for checkpoints) |
| `originals/` | byte-identical copies of the explicit run's `authorized_plan.json`, `preflight.json`, `launch.cmd`, `launch_record.json`, `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `train_actor_gradient_diagnostics.jsonl`, `episode_failures.jsonl`, `invocation_start_local.txt`, `native_exit_code.txt` |
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
python research_evidence/generalized_v2/explicit_ego_readout_ctde_fd100_r1/scripts/extract_evidence.py --check
python research_evidence/generalized_v2/explicit_ego_readout_ctde_fd100_r1/scripts/extract_evidence.py --from-derived
```

`--check` reads all three original run directories, requires byte identity of every generated
file and runs 77 integrity checks. `--from-derived` needs no run directory. All quantities are
descriptive and non-counterfactual.
