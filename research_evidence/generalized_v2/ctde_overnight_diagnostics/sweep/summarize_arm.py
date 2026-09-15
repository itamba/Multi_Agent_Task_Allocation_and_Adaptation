"""PO3 descriptive diagnostic extraction (read-only). Usage: python summarize_arm.py <dir_with_run_summary_eval_train_records>"""
import json, os, statistics, sys

d = sys.argv[1]
S = json.load(open(os.path.join(d, "run_summary.json"), encoding="utf-8"))
ev = [json.loads(l) for l in open(os.path.join(d, "eval_records.jsonl"), encoding="utf-8") if l.strip()]
tr = [json.loads(l) for l in open(os.path.join(d, "train_records.jsonl"), encoding="utf-8") if l.strip()]

ident = S["final_eval_selection"]["identity"]
fin = [r for r in ev if all(r[k] == v for k, v in ident.items())]
assert len(fin) == 1, ident
F = fin[0]
B = F["v2_behaviour"]


def r6(x):
    return None if x is None else round(x, 6)


def summ(key, recs=tr):
    xs = [r[key] for r in recs if r.get(key) is not None]
    if not xs:
        return None
    out = {"first": r6(xs[0]), "last": r6(xs[-1]), "mean": r6(statistics.fmean(xs)),
           "min": r6(min(xs)), "max": r6(max(xs)), "final5_mean": r6(statistics.fmean(xs[-5:])),
           "final25_mean": r6(statistics.fmean(xs[-25:]))}
    return out


def addc(dst, src):
    for k, v in (src or {}).items():
        dst[k] = dst.get(k, 0) + v


tx = [r["n_transitions"] for r in tr]
fd_mild, fd_sev = {}, {}
for r in tr:
    addc(fd_mild, r.get("fd_meta_action_counts_mild"))
    addc(fd_sev, r.get("fd_meta_action_counts_severe"))
ft = S["fuel_damage_totals"]
nmw = sum(r.get("n_mild_fd_wakes", 0) for r in tr)
nsw = sum(r.get("n_severe_fd_wakes", 0) for r in tr)

out = {
    "run": d,
    "final_eval_identity": ident,
    "primary_endpoint": {
        "metric": B["metric"], "macro_mean_over_base_cells": B["macro_mean_over_base_cells"],
        "macro_n_base_cells_defined": B["macro_n_base_cells_defined"],
        "per_cell": {c: v["severe_minus_mild_abort_mass_mean"] for c, v in B["by_base_cell"].items()},
        "directional_switch": [B["directional_switch_count"], B["directional_switch_rate"]],
        "reverse_switch": [B["reverse_switch_count"], B["reverse_switch_rate"]],
        "n_groups_metric_eligible": B["n_groups_metric_eligible"], "n_groups_complete": B["n_groups_complete"],
        "pooled_mean_over_groups": B["pooled_mean_over_groups"],
        "p_abort_mild_mean": r6(statistics.fmean(g["p_abort_mild"] for g in B["groups"] if g["metric_eligible"])),
        "p_abort_severe_mean": r6(statistics.fmean(g["p_abort_severe"] for g in B["groups"] if g["metric_eligible"])),
    },
    "final_eval": {
        "n": [F["n_successful"], F["n_attempted"]],
        "reward_mean": {k: r6(F[f]) for k, f in [("overall", "eval_reward_mean"), ("clean", "eval_reward_mean_clean"),
                                                  ("mild", "eval_reward_mean_mild"), ("severe", "eval_reward_mean_severe")]},
        "deltas": {k: r6(F[k]) for k in ["eval_delta_mild_minus_clean", "eval_delta_severe_minus_clean", "eval_delta_severe_minus_mild"]},
        "meta_action_counts": F["meta_action_counts"],
        "fd_mild": [F["eval_fd_meta_action_counts_mild"], F["eval_fd_meta_action_rates_mild"], F["eval_n_mild_fd_wakes"]],
        "fd_severe": [F["eval_fd_meta_action_counts_severe"], F["eval_fd_meta_action_rates_severe"], F["eval_n_severe_fd_wakes"]],
        "deaths": F["eval_deaths"], "rtb_issued": F["eval_fuel_damage_rtb_issued"],
        "targets_confirmed_unique_mean": r6(F["eval_targets_confirmed_unique_mean"]),
    },
    "eval_trajectory": [(r["evaluation_stage"], r["updates_completed"], r6(r["eval_reward_mean"]),
                         r6(r["eval_reward_mean_mild"]), r6(r["eval_reward_mean_severe"]),
                         r["v2_behaviour"]["macro_mean_over_base_cells"], r["eval_deaths"],
                         r["meta_action_counts"].get("SELF_PRESERVATION_ABORT")) for r in ev],
    "learning": {
        "updates": S["updates_completed"], "total_transitions": S["total_transitions"],
        "transitions_per_update": {"min": min(tx), "median": statistics.median(tx), "mean": r6(statistics.fmean(tx)), "max": max(tx)},
        **{k: summ(k) for k in ["train_reward_mean", "policy_loss", "entropy", "approx_kl", "clip_fraction",
                                "grad_norm", "value_loss", "value_mean", "value_target_mean", "critic_grad_norm"]},
        "run_seconds": S["run_seconds"],
    },
    "training_population": {
        k: ft[k] for k in ["train_clean_successful", "train_mild_successful", "train_severe_successful",
                           "train_clean_failed", "train_mild_failed", "train_severe_failed",
                           "train_fuel_damage_events_applied", "train_fuel_damage_wakes",
                           "train_fuel_damage_rtb_issued", "train_deaths"]
    } | {"train_attempted": S["train_episodes_attempted"], "train_failed": S["train_episodes_failed"],
         "replacement_attempts": S["train_replacement_attempts"], "failures_by_error_type": S["failures_by_error_type"],
         "train_meta_action_totals": S["meta_action_totals"],
         "fd_mild_counts": fd_mild, "fd_mild_wakes": nmw,
         "fd_mild_rates": {k: r6(v / nmw) for k, v in fd_mild.items()} if nmw else None,
         "fd_severe_counts": fd_sev, "fd_severe_wakes": nsw,
         "fd_severe_rates": {k: r6(v / nsw) for k, v in fd_sev.items()} if nsw else None},
}
print(json.dumps(out, indent=1, default=str))
