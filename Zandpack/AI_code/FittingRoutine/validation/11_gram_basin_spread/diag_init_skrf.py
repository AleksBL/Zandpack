"""Final hypothesis for the Ag3Au3 exactera cache irreproducibility:
the cached fit (10-08 13:55) may predate the psd -> psd_dict_init
routing (the package checkpoint at 10-08 19:51 already has it). If so,
the cached fit was initialized from SKRF relocation poles while every
current run starts from the NNLS PSD dictionary.

Test: seed the production exact fit with skrf_init poles (bypassing
the dictionary), everything else identical to BASE_KW. If this
reproduces rms 0.0171417 / 94 poles, the cache era is explained: the
INIT changed, not the solver or the knobs.
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
    fd = lf.prepare(E, F)
    # the pre-dictionary era init: skrf relocation (the dispatch's
    # psd=False path, e_margin 0.15 = fit() default)
    c0, w0 = lf.init_poles(fd.E, fd.F, 24, method="skrf", e_margin=0.15)
    print(f"[skrf-init] {c0.size} skrf poles", flush=True)

    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0, tr_solver="exact",
                   init_poles=(c0, w0))
    t0 = time.perf_counter()
    r = lf.fit_boosted(E, F, BASE_KW, n_res=8)
    wall = time.perf_counter() - t0
    print(f"[skrf-init] exact fit from skrf poles: rms {r.rms:.6g} "
          f"n={r.n_poles} wall {wall:.0f}s", flush=True)
    print("[skrf-init] targets: exactera cache rms 0.0171417 n=94 | "
          "dict-init exact c0 rms 0.0258409 n=108", flush=True)


if __name__ == "__main__":
    main()
