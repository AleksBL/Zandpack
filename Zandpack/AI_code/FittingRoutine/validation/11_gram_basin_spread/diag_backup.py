"""Diagnostic 2: which code-era change moved the Ag3Au3 exact baseline?
Run the natural exact fit with a BACKUP version of AI_fit (copied to
/tmp) to bisect the rounds between the exactera cache (2026-10-08
13:55) and now.

Usage: python3 diag_backup.py <backup.py> [label]
OMP_NUM_THREADS=1. ~15 min per run.
"""
import gzip
import importlib.util
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spread_common as sc  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    bk = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.basename(bk).replace(".py", "")
    os.chdir("/tmp")
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R = pickle.load(f)
    lf = load_module(bk, "ai_fit_backup")

    E, F = lf.extract_dense(R, 1, 0)
    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0)
    kw = dict(BASE_KW)
    if hasattr(lf.fit, "__code__") and "tr_solver" in \
            lf.fit.__code__.co_varnames:
        kw["tr_solver"] = "exact"
    t0 = time.perf_counter()
    r = lf.fit_boosted(E, F, kw, n_res=8)
    wall = time.perf_counter() - t0
    print(f"[bk:{label}] natural exact fit_boosted: rms {r.rms:.6g} "
          f"n={r.n_poles} wall {wall:.0f}s", flush=True)
    print("[bk] targets: exactera cache rms 0.0171417 n=94 | "
          "current seeded c0 rms 0.0258409 n=108", flush=True)


if __name__ == "__main__":
    main()
