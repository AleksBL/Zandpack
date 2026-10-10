"""Check whether the Ag3Au3 exactera cached fits were made on the SAME
data as today's extraction (block-storage fix 4.7.1 landed around the
cache's creation date): re-evaluate each cached fit on today's
extract_dense output and compare to its stored rms attribute.

If the re-evaluated rms >> stored rms, the cache was fit to different
INPUT data (e.g. halved lower-triangle blocks), explaining why no code
version reproduces it. If they match, the cache was fit to the same
data with different fit PARAMETERS.
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
    print("stored block inds:", list(R.Nonortho_Gammas[0].inds),
          flush=True)
    sys.path.insert(0, sc.ROOT)
    import refine_sampling  # noqa: F401  (unpickling shim)
    sys.path.remove(sc.ROOT)
    d = np.load(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                            "cmp_fits_exactera.npz"),
                allow_pickle=True)
    for lead in (0, 1):
        fit = d[f"lead{lead}_ik0"].item()
        E, F = refine_sampling.extract_dense(R, lead, 0)
        D = fit.evaluate(E) - F
        # fit_boosted's comb.rms is the RELATIVE frobenius rms
        # (frobenius_rms_error: sqrt(sum|diff|^2 / sum|F|^2))
        rel = float(np.sqrt(np.sum(np.abs(D) ** 2)
                            / np.sum(np.abs(F) ** 2)))
        print(f"lead{lead}: stored rms attr {fit.rms:.10g} | "
              f"RELATIVE rms on TODAY's extract: {rel:.10g}",
              flush=True)


if __name__ == "__main__":
    main()
