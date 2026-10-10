"""Full inspection of the Ag3Au3 exactera cached fits (user point 2):
every stored attribute, plus pole statistics that fingerprint the fit
config - the width floor (eta_floor_mult x eta_raw), the width ceiling
(w_max_raw), center range (window / e_margin), and pole counts.

Current production knobs for Ag3Au3 would predict: w in [0.05, 0.5]
(mult=5, eta=1e-2, w_max=0.5), centers within the sampling window
(+/-15% margin), base+boost counts below the 100 cap.
"""
import gzip
import os
import pickle
import sys

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
    sys.path.insert(0, sc.ROOT)
    import refine_sampling  # noqa: F401
    sys.path.remove(sc.ROOT)

    E0, _ = lf.extract_dense(R, 0, 0)
    E1, _ = lf.extract_dense(R, 1, 0)
    print(f"sampling grid: lead0 {E0.size} pts [{E0.min():.3f}, "
          f"{E0.max():.3f}], lead1 {E1.size} pts", flush=True)

    d = np.load(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                            "cmp_fits_exactera.npz"),
                allow_pickle=True)
    for lead in (0, 1):
        fit = d[f"lead{lead}_ik0"].item()
        print(f"\n--- lead{lead}: {type(fit).__name__} ---", flush=True)
        attrs = [a for a in vars(fit)]
        print("stored attrs:", attrs, flush=True)
        print(f"n_poles={fit.n_poles}  rms attr={fit.rms:.6g}  "
              f"history={dict(fit.history)}", flush=True)
        w = np.asarray(fit.w)
        e = np.asarray(fit.e)
        C = np.asarray(fit.C)
        print(f"widths : min {w.min():.5f}  max {w.max():.5f}  "
              f"median {np.median(w):.5f}", flush=True)
        print(f"centers: min {e.min():.4f}  max {e.max():.4f}  "
              f"(window [{E1.min():.3f}, {E1.max():.3f}], "
              f"e_margin +-15% = +-"
              f"{0.15 * (E1.max() - E1.min()):.2f})", flush=True)
        print(f"residues: |C| max {np.abs(C).max():.4g}  "
              f"n nonzero-ish {int((np.abs(C) > 1e-12 * np.abs(C).max()).sum())}"
              f" of {C.size}", flush=True)
        # config fingerprints
        if w.min() > 0.049:
            print("-> min width consistent with eta_floor_mult=5 "
                  "(floor 0.05)", flush=True)
        elif w.min() > 0.009:
            print(f"-> min width {w.min():.4f} < 0.05: mult was "
                  f"LOWER than 5 at fit time", flush=True)
        if w.max() > 0.5:
            print(f"-> max width {w.max():.4f} EXCEEDS w_max=0.5: "
                  f"no width cap at fit time", flush=True)


if __name__ == "__main__":
    main()
