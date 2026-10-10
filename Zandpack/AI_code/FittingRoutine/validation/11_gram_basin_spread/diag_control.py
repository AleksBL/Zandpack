"""Diagnostic: why does the seeded control (init_poles = the dispatch
poles) not reproduce the exact-era Ag3Au3 cache (0.0171 / 94 poles)?

Run the NATURAL path (no init_poles) with the identical apply_method
kwargs, tr_solver='exact', on the current AI_fit. Three outcomes:
- rms/n == 0.02584/108  -> the seeded control IS the natural current
  path; the cache predates a code-era change and is no longer
  reproducible (an era finding, not a harness bug).
- rms/n == 0.0171/94    -> init_poles seeding diverges from the
  internal dispatch; the harness needs fixing.
- anything else         -> the natural path itself is unstable
  run-to-run (BLAS basin noise).

OMP_NUM_THREADS=1. ~15 min.
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


def main():
    os.chdir("/tmp")
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R = pickle.load(f)
    lf = sc.import_ai_fit()

    E, F = lf.extract_dense(R, 1, 0)
    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0, tr_solver="exact")
    t0 = time.perf_counter()
    r = lf.fit_boosted(E, F, BASE_KW, n_res=8)
    wall = time.perf_counter() - t0
    print(f"[diag] natural exact fit_boosted: rms {r.rms:.6g} "
          f"n={r.n_poles} wall {wall:.0f}s", flush=True)
    print("[diag] exactera cache:            rms 0.0171417 n=94",
          flush=True)
    print("[diag] seeded control (c0):       rms 0.0258409 n=108",
          flush=True)

    # also verify the dispatch poles are identical when recomputed
    # (determinism of psd_dict_init across the two call sites)
    c0, w0, _ = sc.baseline_init_poles(lf, E, F)
    c1, w1, _ = sc.baseline_init_poles(lf, E, F)
    print(f"[diag] dispatch determinism: "
          f"e bit-equal {np.array_equal(c0, c1)}, "
          f"w bit-equal {np.array_equal(w0, w1)}, "
          f"n={c0.size}", flush=True)


if __name__ == "__main__":
    main()
