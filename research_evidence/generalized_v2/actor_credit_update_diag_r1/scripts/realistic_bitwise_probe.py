"""Post-launch engineering probe (synthetic data only; no BLADE, no solver, no episode).

Question: on REALISTIC graph sizes (k up to 14 task nodes, several agents, ~30 transitions per
batch), are the BASE updater (compiled from the verified base commit), the OFF updater and the ON
updater bit-identical in one process, and is the OFF update bit-identical across two separate
processes? Prints a JSON line with per-seed parameter digests; run it twice to compare processes.

Usage: python realistic_bitwise_probe.py <base_sha> [n_seeds]
"""

from __future__ import annotations

import __future__
import ast
import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch

from match_aou.rl.action.graph_action import build_action_mask, sample_action
from match_aou.rl.observation.graph_builder import EdgeType, GraphObservation
from match_aou.rl.training import graph_ppo as GP
from match_aou.rl.training import graph_tick_loop as TL
from match_aou.rl.training.graph_ppo import EpisodeRecord, PPOConfig, PPOUpdater


def base_class(sha):
    src = subprocess.run(["git", "show", "%s:src/match_aou/rl/training/graph_ppo.py" % sha],
                         capture_output=True, text=True, check=True).stdout
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "PPOUpdater")
    ns = dict(vars(GP))
    exec(compile(ast.get_source_segment(src, node), "<base>", "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), ns)
    return ns["PPOUpdater"]


def obs(rng, k, a):
    tf = rng.random((k, 6)).astype(np.float32)
    tf[:, 2] = 1.0
    tf[:, 3] = (rng.random(k) > 0.3).astype(np.float32)
    tf[:, 5] = (rng.random(k) > 0.4).astype(np.float32)
    af = np.zeros((a, 2), dtype=np.float32)
    af[0] = rng.random(2).astype(np.float32) * [1.0, 2.0] - [0.0, 0.5]
    own = sorted(rng.choice(k, size=max(1, k // 3), replace=False).tolist())
    peers = [j for j in range(k) if j not in own and rng.random() < 0.5]
    src = [k] * len(own) + [k + 1 + (j % (a - 1)) for j in peers]
    dst = own + peers
    return GraphObservation(
        task_features=tf, agent_features=af, ego_index=k,
        edge_index=np.array([src, dst], dtype=np.int64),
        edge_type=np.full((len(src),), int(EdgeType.ASSIGNMENT), dtype=np.int64),
        task_target_ids=["t%d" % i for i in range(k)], agent_ids=["a%d" % i for i in range(a)],
        agent_id="a0", current_time=100, time_norm=float(rng.random()))


def records(policy, seed):
    rng = np.random.default_rng(seed)
    out = []
    for e in range(8):
        traj = []
        for t in range(int(rng.integers(1, 6))):
            g = obs(rng, int(rng.integers(3, 15)), int(rng.integers(2, 7)))
            with torch.no_grad():
                torch.manual_seed(seed * 100 + e * 10 + t)
                m, n, lp, ent = sample_action(policy.head(policy.encoder(g)),
                                              build_action_mask(g))
            tr = TL.Transition(gobs=g, ego_id="e%d" % (t % 2), tick=t, meta_action=int(m),
                               node_v=n, log_prob=float(lp.item()), entropy=float(ent.item()))
            tr.wake_kind = TL.WAKE_KIND_ORDINARY
            traj.append(tr)
        r = -float(rng.random())
        traj[-1].reward = r
        out.append(EpisodeRecord.from_trajectory(traj, r, seed=seed + e, episode_index=e))
    return out


def digest(policy, upd):
    h = hashlib.sha256()
    for p in list(policy.encoder.parameters()) + list(policy.head.parameters()):
        h.update(p.detach().numpy().tobytes())
    for st in upd.optimizer.state_dict()["state"].values():
        for v in st.values():
            h.update(v.numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()[:16]


def main():
    sha = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    Base = base_class(sha)
    out = {"torch": torch.__version__, "threads": torch.get_num_threads(), "seeds": {}}
    for seed in range(n_seeds):
        torch.manual_seed(3000000 + seed)
        p0 = TL.build_policy()
        recs = records(p0, seed)
        n = sum(r.n_transitions for r in recs)
        res = {"n_transitions": n}
        for name, cls, on in (("base", Base, False), ("off", PPOUpdater, False),
                              ("on", PPOUpdater, True)):
            p = copy.deepcopy(p0)
            u = cls(p, PPOConfig())
            for _ in range(2):
                kw = ({"step_group_ids": [i % 4 for i in range(n)], "step_sink": lambda r: None,
                       "step_contrast_ids": (1, 0)} if on else {})
                u.update(recs, **kw)
            res[name] = digest(p, u)
        res["base_eq_off"] = res["base"] == res["off"]
        res["off_eq_on"] = res["off"] == res["on"]
        out["seeds"][seed] = res
    print(json.dumps(out))


if __name__ == "__main__":
    main()
