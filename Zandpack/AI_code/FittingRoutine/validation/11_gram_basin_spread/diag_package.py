"""Diagnostic 3: does the PACKAGE-era code (checkpoint_2026-10-08_
lorentzfit: lorentz_fit/ + the package-era refine_sampling driver, the
era that created the Ag3Au3 exactera cache at 2026-10-08 13:55)
reproduce the cached fit (rms 0.0171417 / 94 poles)?

If YES: the divergence entered during the package -> single-file
merge (2026-10-08 evening) or a later round - a real code-path
difference to flag for the migration.
If NO (0.0258409/108 again): the cache was created with a different
CONFIG, not a different code - the stored numbers then correspond to
settings we are no longer using.

OMP_NUM_THREADS=1. ~15-30 min.
"""
import gzip
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spread_common as sc  # noqa: E402

PKG = os.path.join(sc.ROOT, "checkpoint_2026-10-08_lorentzfit")


def main():
    os.chdir("/tmp")
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R = pickle.load(f)

    sys.path.insert(0, PKG)
    import refine_sampling as rs_pkg
    sys.path.remove(PKG)

    E, F = rs_pkg.extract_dense(R, 1, 0)
    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0)
    t0 = time.perf_counter()
    r = rs_pkg.fit_boosted(E, F, BASE_KW, n_res=8)
    wall = time.perf_counter() - t0
    print(f"[pkg] package-era fit_boosted: rms {r.rms:.6g} "
          f"n={r.n_poles} wall {wall:.0f}s", flush=True)
    print("[pkg] targets: exactera cache rms 0.0171417 n=94 | "
          "current/merged rms 0.0258409 n=108", flush=True)


if __name__ == "__main__":
    main()
