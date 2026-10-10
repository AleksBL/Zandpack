"""Cross-system spot check after the refit_scales optimization:
RSSE lead0, gram@1e-9, control start, production kwargs (the spread
round recorded c0: rms 0.008541 / 45 poles, wall ~8 s)."""
import lzma
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, os.path.join(ROOT, "validation",
                                "11_gram_basin_spread"))
import spread_common as sc  # noqa: E402


def main():
    os.chdir(os.path.join(ROOT, "Data2", "VerySmallRSSE", "refined"))
    with lzma.open("C1.xz", "rb") as f:
        R = pickle.load(f)
    lf = sc.import_ai_fit()
    E, F = lf.extract_dense(R, 0, 0)
    c0, w0, _ = sc.baseline_init_poles(lf, E, F)
    kw = dict(n_init=24, max_poles=150, target_rms=5e-4,
              psd=True, psd_force=False, eta_raw=1e-2,
              local_floor_frac=1.5, w_max_raw=0.5,
              eta_floor_mult=3.0, tr_solver="gram", gram_cut=1e-9,
              init_poles=(c0, w0))
    t0 = time.perf_counter()
    r = lf.fit_boosted(E, F, kw, n_res=8)
    wall = time.perf_counter() - t0
    print(f"[rsse-spot] gram c0: rms {r.rms:.6g} n={r.n_poles} "
          f"wall {wall:.1f}s "
          f"(spread c0 record pre-optimization: 0.008541/45, ~8 s)",
          flush=True)


if __name__ == "__main__":
    main()
