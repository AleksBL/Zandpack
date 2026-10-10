"""Per-block analysis of the Ag3Au3 exactera cached lead1 fit on
today's extraction: block magnitudes and per-block fit residuals,
plus the deliberately HALVED-off-diagonal variant. Determines which
data convention the cache was actually fit to (full blocks vs the
pre-4.7.1 halved assembly), given that its stored rms attr (0.0171)
disagrees with today's re-evaluation (0.0025).
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

    # raw block assembly (no partner completion): which blocks exist
    G = R.Nonortho_Gammas[1]
    print("lead1 block inds:", [tuple(b) for b in G.inds], flush=True)
    sidx = R.sampling_idx[1]
    blocks = {}
    for b, (i, j) in enumerate(G.inds):
        arr = lf.bs2np(G)[b] if hasattr(lf, "bs2np") else None
        blocks[(i, j)] = arr
    # use extract_dense for the dense data
    E, F = lf.extract_dense(R, 1, 0)
    p = F.shape[1]
    # infer block sizes from the stored Gamma blocks
    n0 = blocks[(0, 0)].shape[-1] if (0, 0) in blocks else None
    if n0 is None:
        # fallback: equal split
        n0 = p // 2
    n1 = p - n0
    sl = {(0, 0): (slice(0, n0), slice(0, n0)),
          (0, 1): (slice(0, n0), slice(n0, p)),
          (1, 0): (slice(n0, p), slice(0, n0)),
          (1, 1): (slice(n0, p), slice(n0, p))}

    d = np.load(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                            "cmp_fits_exactera.npz"),
                allow_pickle=True)
    fit = d["lead1_ik0"].item()
    Ffit = fit.evaluate(E)
    res = np.abs(Ffit - F)
    for key in sl:
        a, b = sl[key]
        m = F[:, a, b]
        r = res[:, a, b]
        print(f"block {key}: |F| mean {np.abs(m).mean():8.4g} "
              f"max {np.abs(m).max():8.4g} | residual mean "
              f"{r.mean():8.4g} max {r.max():8.4g}", flush=True)
    # halved-off-diagonal variant (pre-4.7.1 assembly)
    Fhalf = F.copy()
    for key in [(0, 1), (1, 0)]:
        a, b = sl[key]
        Fhalf[:, a, b] = 0.5 * F[:, a, b]
    rms_half = float(np.sqrt(np.mean(np.abs(Ffit - Fhalf) ** 2)))
    print(f"cached fit rms vs HALVED-offdiag variant: {rms_half:.6g} "
          f"(stored attr 0.0171, vs today's full data 0.0025)",
          flush=True)


if __name__ == "__main__":
    main()
