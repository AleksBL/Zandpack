"""ag3au3 follow-up: (a) gram identity + step agreement on REAL ag3au3
factors (where cond(J) is expected much higher than RSSE); (b) gram_cut
sweep on the full production fit (each gram run ~45s): does the basin
degradation depend on the truncation floor?"""
import gzip
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import svd
from scipy.optimize._lsq.common import solve_lsq_trust_region

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'Ag3Au3'))
with gzip.open("tmp.gz", "rb") as f:
    R = pickle.load(f)          # pickles first
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
from refine_sampling import extract_dense, fit_boosted  # noqa: E402
sys.path.remove(ROOT)

E, F = extract_dense(R, 1, 0)
fd = lf.prepare(E, F)
chans = lf.hermitian_channels(fd.F.shape[1])
B = lf.pack_responses(fd.F, chans)
sw = lf._sample_weights(fd.F, "relative", quad=lf._quad_widths(fd.E))
Bc = lf.compress_channels(B, 1e-9)

print(f"p={fd.F.shape[1]} n_ch={B.shape[0]} -> compressed {Bc.shape[0]}",
      flush=True)

# --- (a) identity + step agreement at a late-ish iterate ---
e0, w0 = lf.linspace_init(fd.E, 40)
r0 = lf.fit_poles(fd.E, Bc, e0, w0, sample_weights=sw, ridge=0.05,
                  bounds_fun=lf.default_bounds(fd.E), max_nfev=60,
                  tr_solver="lsmr")
th = np.empty(2 * r0["e"].size)
th[0::2] = r0["e"]
th[1::2] = r0["w"]

core = lf._varproj_core(fd.E, Bc, th, sw, None, ridge=0.05)
Z, K, S = lf._varproj_jac(core, fd.E, sw, None)
Res, X = core["Res"], core["X"]
J = lf._assemble_dense_j(Res, X, Z, K, S)
f = Res.ravel(order="C")
if core["svd_path"]:
    G, g = lf.gram_and_grad(Res, X, Z, S, K=K)
else:
    G, g = lf.gram_and_grad(Res, X, Z, S, R1=core["R1"], Q=core["Q"])
errG = np.linalg.norm(G - J.T @ J) / np.linalg.norm(J.T @ J)
errg = np.linalg.norm(g - J.T @ f) / np.linalg.norm(J.T @ f)
sv = np.linalg.svd(J, compute_uv=False)
cond = sv[0] / sv[-1]
print(f"(a) identity G err {errG:.2e}  g err {errg:.2e}  cond(J) {cond:.2e}  "
      f"dirs<1e-7*s0: {int(np.sum(sv < 1e-7 * sv[0]))}/{sv.size}  "
      f"dirs<1e-12*s0: {int(np.sum(sv < 1e-12 * sv[0]))}/{sv.size}",
      flush=True)

# --- (b) gram_cut sweep on the production fit ---
CFG = dict(n_init=24, max_poles=100, target_rms=2e-3,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=5.0, channel_tol=1e-4)
import time
for cut in (1e-7, 1e-9, 1e-11):
    kw = dict(CFG)
    kw['tr_solver'] = 'gram'
    kw['gram_cut'] = cut
    t0 = time.time()
    r = fit_boosted(E, F, kw, n_res=8)
    wall = time.time() - t0
    Eg = np.linspace(E.min(), E.max(), 801)
    me = lf.psd_min_eigenvalue(Eg, r.e, r.w, r.C)
    print(f"(b) gram_cut={cut:.0e}: wall {wall:6.1f}s  rms {r.rms:.6g}  "
          f"n={r.n_poles}  min_eig {me:.3e}", flush=True)
print("DONE")
