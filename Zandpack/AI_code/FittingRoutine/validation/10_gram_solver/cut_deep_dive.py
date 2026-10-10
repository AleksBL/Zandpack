"""never_form_J follow-up (user points 1+2): WHERE does the cut matter,
and are the truncated directions REAL or eigh noise?

Runs the FULL production ag3au3 lead1 fit twice (gram, cut=1e-7 and
1e-9) with the documented AI_fit diagnostic hook (gram_diag_hook)
active. At EVERY outer iteration:
  - log min(s)/s[0] of the augmented Gram spectrum and n_trunc.
At every iteration where truncation triggers (n_trunc > 0) or the
spectrum is weak (s_min/s0 < 1e-6):
  - rebuild the dense J from the hook record's factor cache (context:
    Res/X/Q/R1 + Z/K/S; K recomputed from R1/Q on the QR path where
    the gram branch skips it), form J_aug = [J*d; diag(sqrt(diag_h))]
    and SVD it, and compare the weak end against eigh(G_aug):
      * paired-by-rank singular values |s_i - sv_i| / sv_i (weakest 3)
      * eigenvector overlaps |<v_eigh, v_svd>|
eigh has absolute eigenvalue error ~ eps * lam_max, so singular
values below ~1e-8*s0 are unreliable: if svd(J_aug) DISAGREES with
eigh there, the cut=1e-9 benefit is incidental; if they agree, the
weak directions are real.
OMP_NUM_THREADS=1.
"""
import gzip
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import solve_triangular, svd

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'Ag3Au3'))
with gzip.open("tmp.gz", "rb") as f:
    R = pickle.load(f)          # pickles first
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
from refine_sampling import extract_dense, fit_boosted  # noqa: E402
sys.path.remove(ROOT)

E, F = extract_dense(R, 1, 0)
CFG = dict(n_init=24, max_poles=100, target_rms=2e-3,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=5.0, channel_tol=1e-4)

records = []
trigger_dumps = []


def hook(rec):
    s = rec["s"]
    s0 = s[0]
    ratio = float(s[-1] / s0) if s0 > 0 else 0.0
    ctx = rec.get("context")
    core = ctx["core"] if ctx is not None else None
    records.append(dict(cost=rec["cost"], smin_ratio=ratio,
                        n_trunc=rec["n_trunc"], M=rec["n"] // 2))
    # cross-check at triggers / weak spectra
    if (rec["n_trunc"] > 0 or ratio < 1e-6) and core is not None:
        Z, K, S = ctx["jac"]
        Res, X = core["Res"], core["X"]
        if K is None and core["R1"] is not None:    # QR path, K skipped
            K = solve_triangular(core["R1"], core["Q"].T,
                                 check_finite=False).T
        if K is None:
            return
        J = lf._assemble_dense_j(Res, X, Z, K, S)
        d, diag_h = rec["d"], rec["diag_h"]
        J_aug = np.vstack([J * d, np.diag(np.sqrt(np.maximum(diag_h,
                                                            0.0)))])
        _U, sv2, Vt = svd(J_aug, full_matrices=False)
        sv2 = sv2[::-1]                              # ascending
        Vt = Vt[::-1]                                # rows: weak first
        s_asc = s[::-1]
        pairs = []
        for i in range(min(3, s.size)):
            rel = abs(s_asc[i] - sv2[i]) / max(sv2[i], 1e-300)
            v_eig = rec["V"][:, -1 - i]               # weakest last
            ov = abs(float(v_eig @ Vt[i]))
            pairs.append((float(s_asc[i] / s[0]),
                          float(sv2[i] / sv2.max()), rel, ov))
        trigger_dumps.append(dict(M=rec["n"] // 2, it=rec["iteration"],
                                  n_trunc=rec["n_trunc"], pairs=pairs))


lf.gram_diag_hook = hook

for cut in (1e-7, 1e-9):
    records.clear()
    trigger_dumps.clear()
    kw = dict(CFG)
    kw['tr_solver'] = 'gram'
    kw['gram_cut'] = cut
    t0 = time.time()
    r = fit_boosted(E, F, kw, n_res=8)
    wall = time.time() - t0
    ratios = np.array([rr["smin_ratio"] for rr in records])
    ntr = np.array([rr["n_trunc"] for rr in records])
    n_iter = len(records)
    n_with_trunc = int(np.sum(ntr > 0))
    n_weak = int(np.sum(ratios < 1e-6))
    n_beyond_floor = int(np.sum(ratios < 1e-8))
    print(f"\n=== gram cut={cut:.0e}: wall {wall:.1f}s  rms {r.rms:.6g}  "
          f"n={r.n_poles}  outer-iterations {n_iter} ===", flush=True)
    print(f"    min(s)/s0 over iterations: min {ratios.min():.2e}  "
          f"median {np.median(ratios):.2e}  max {ratios.max():.2e}",
          flush=True)
    print(f"    iterations with n_trunc>0: {n_with_trunc};  "
          f"spectrum < 1e-6: {n_weak};  < 1e-8 (eigh noise floor): "
          f"{n_beyond_floor}", flush=True)
    if n_with_trunc:
        perM = {}
        for rr, t in zip(records, ntr):
            if t > 0:
                perM.setdefault(rr["M"], [0, 0])
                perM[rr["M"]][0] += 1
                perM[rr["M"]][1] += int(t)
        perM_s = "; ".join(f"M={m}: {c} iters, {d} dirs"
                           for m, (c, d) in sorted(perM.items()))
        print(f"    truncation by pole count: {perM_s}  "
              f"(total {int(ntr.sum())})", flush=True)
    print(f"    J-vs-G cross-checks (triggers + weak spectra): "
          f"{len(trigger_dumps)}", flush=True)
    worst_rel = 0.0
    worst_ov = 1.0
    for td in trigger_dumps:
        for (sr_eig, sr_svd, rel, ov) in td["pairs"]:
            worst_rel = max(worst_rel, rel)
            if sr_eig < 1e-6:
                worst_ov = min(worst_ov, ov)
    print(f"    weakest-3 paired s vs svd(J_aug): worst rel diff "
          f"{worst_rel:.2e};  weak-end eigvec overlap min {worst_ov:.2e}",
          flush=True)
    for td in trigger_dumps[:5]:
        ps = ", ".join(f"eig {a:.1e} vs svd {b:.1e} (rel {c:.0e}, "
                       f"ov {d:.2f})" for a, b, c, d in td["pairs"])
        print(f"      M={td['M']} it={td['it']} "
              f"n_trunc={td['n_trunc']}: {ps}", flush=True)

lf.gram_diag_hook = None
print("\nDONE")
