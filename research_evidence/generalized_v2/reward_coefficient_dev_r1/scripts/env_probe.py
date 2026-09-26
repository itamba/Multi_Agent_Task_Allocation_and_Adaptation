"""Environment identity probe for one REWARD-01 arm (no training, solve or evaluation).

Run by ``launch_with_watchdog.py`` under the same conda env, PYTHONPATH and environment block
as the training process it is about to start. Writes package versions and paths, the device,
and the thread settings a fresh torch process sees by default (the trainer sets none).

Usage: python env_probe.py <out.json>
"""

import json
import os
import platform
import sys
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "KMP_AFFINITY",
               "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONNOUSERSITE",
               "CONDA_DEFAULT_ENV")


def _pkg(name):
    try:
        mod = __import__(name)
        return {"version": getattr(mod, "__version__", None),
                "path": str(Path(mod.__file__).resolve())}
    except Exception as exc:  # recorded, never hidden
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def main():
    import torch
    rec = {
        "record": "env_probe", "record_version": 1,
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {n: _pkg(n) for n in ("torch", "numpy", "scipy", "pyomo", "gymnasium",
                                          "shapely", "haversine", "blade", "match_aou")},
        "torch": {"num_threads": torch.get_num_threads(),
                  "num_interop_threads": torch.get_num_interop_threads(),
                  "cuda_available": torch.cuda.is_available(),
                  "default_dtype": str(torch.get_default_dtype()),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "parallel_info": torch.__config__.parallel_info()},
        "device": "cpu",
        "env": {k: os.environ.get(k) for k in THREAD_VARS},
    }
    Path(sys.argv[1]).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("env_probe written", sys.argv[1])


if __name__ == "__main__":
    main()
