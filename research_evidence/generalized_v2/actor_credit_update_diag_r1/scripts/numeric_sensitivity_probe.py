"""Post-launch engineering probe (synthetic data only): is the OFF update's float result sensitive
to the intra-op thread count or to heap allocation / alignment shifts in the same process?
Reuses realistic_bitwise_probe's records. Prints one JSON line of parameter digests.
Usage: python numeric_sensitivity_probe.py [n_seeds]"""
import copy, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import realistic_bitwise_probe as P
from match_aou.rl.training import graph_tick_loop as TL
from match_aou.rl.training.graph_ppo import PPOConfig, PPOUpdater

def run(p0, recs, threads=None, junk=0):
    if threads: torch.set_num_threads(threads)
    hold = [torch.empty(1 + 37 * i) for i in range(junk)]
    p = copy.deepcopy(p0); u = PPOUpdater(p, PPOConfig())
    for _ in range(2): u.update(recs)
    del hold
    return P.digest(p, u)

out = {}
base_threads = torch.get_num_threads()
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    torch.manual_seed(3000000 + seed); p0 = TL.build_policy(); recs = P.records(p0, seed)
    r = {"t%d" % t: run(p0, recs, threads=t) for t in (1, 2, 3, base_threads)}
    torch.set_num_threads(base_threads)
    r.update({"junk%d" % j: run(p0, recs, junk=j) for j in (0, 3, 50, 501)})
    r["thread_sensitive"] = len({r["t%d" % t] for t in (1, 2, 3, base_threads)}) > 1
    r["alignment_sensitive"] = len({r[k] for k in r if k.startswith("junk")}) > 1
    out[seed] = r
print(json.dumps(out))
