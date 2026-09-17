# Semantic-action CTDE actor-gradient development diagnostics — compact evidence index

**Development evidence only; not confirmatory.** This is a compact Git index for two completed
150-update CTDE development diagnostic runs. The review record, interpretation and non-claims
are in [`docs/history/measurements.md` §11](../../../docs/history/measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics).

## Authority of the external originals

The **original run directories are the authoritative artifacts**. They are external to Git,
were read only, and are **not modified, moved, regenerated or copied** by this package. No
record stream, checkpoint, scenario or plot is committed here; each is identified by SHA-256
and byte size in `artifact_manifest.json` (scenarios and plots by a tree digest).

| Run | Directory | Question |
|---|---|---|
| A — p = 0.5 diagnostic baseline | `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_r1_seed3000000_6ed964a` | do immediate-FD transitions produce actor-gradient pressure toward larger `SEVERE − MILD P(ABORT)`, and does the rest of the batch cancel it? |
| B — FD100 intervention | `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` | does raising training FD exposure `0.5 → 1.0` change that signal and held-out behaviour? |

## Identity

- **Measured code SHA (both runs):** `6ed964a1abd09de2130aee3d0d314c8f32165056`, branch
  `task/v2-ctde-gradient-pressure-diagnostics` — the GPT-approved head of PR #74, **unmerged
  when measured**; `run_config.json:/provenance/git` records `dirty = false`, 0 dirty paths.
- **Design:** GENERALIZED-V2, `ctde`, `p1_milp_v1`, `semantic_k_plus_2_logmeanexp_v1`, DEVELOPMENT
  profile of manifest `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`,
  base seed `3000000`, 150 updates × 8 successful episodes, `eval_every = 25`,
  `--actor-gradient-diagnostics` on, early stopping off.
- **Comparison boundary:** the only scientific configuration intervention in B versus A is
  `fuel_damage_probability 0.5 → 1.0` (plus a fresh `output_dir`). The flattened `run_config.json`
  difference, listed in `diagnostic_summary.json:/run_config_paths_differing_B_vs_A`, is exactly
  that key, its derived `difficulty` block (FD probability and scheduled cell probabilities), the
  output directory, the invocation argv and the collection timestamp. **Known BLADE run-to-run
  timing nondeterminism means the two runs' physical trajectories are not claimed bit-identical**
  apart from the intervention.
- **Verdicts:** both `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` — GPT orchestrator
  review of 2026-09-17, transferred through the user-approved documentation packet.

## Files

| File | Content |
|---|---|
| `artifact_manifest.json` | SHA-256 and bytes of every top-level run file and each checkpoint; file count, bytes and tree digest of `checkpoints/`, `plots/`, `scenarios/` |
| `diagnostic_summary.json` | completion and accounting counts, provenance, the seven-round evaluation trajectory, gradient 25-update windows with denominators, entropy and reconstruction checks, credit quantities, and the B-vs-A configuration difference |
| `scripts/extract_summary.py` | standard-library extractor that produced both JSON files from the external originals |

Reproduce or verify from the repository root (reads only; `--check` writes nothing):

```bash
python research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/scripts/extract_summary.py --check
```

Definitions used by the summary (medians are `statistics.median`):

- gradient windows and signed counts use only updates whose `separation_contrast` is defined
  (both MILD and SEVERE immediate-FD transitions present);
- credit population: `immediate_fuel_damage` rows with `measurement_join.is_fd_selected_ego = true`;
  `ABORT − PLAN` is the mean normalized advantage of rows selecting
  `SELF_PRESERVATION_ABORT` minus that of rows selecting `PLAN_COMPLIANCE`, within severity;
  within-update `SEVERE − MILD` is `mean(severe) − mean(mild)` per update holding both, then the
  median over those updates; the TD ratio is the per-row median of
  `|td_residual| / |raw_advantage|`. All are descriptive and non-counterfactual.

This package is an index committed on the task branch, not a separate evidence-only commit on a
measured SHA ([`experiments.md` §5](../../../docs/workflows/experiments.md#5-evidence-preservation)
names that shape as one adequate option, not a universal requirement).
