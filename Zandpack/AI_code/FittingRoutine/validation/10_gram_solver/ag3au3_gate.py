"""never_form_J guide, validation step 6 on the HEAVY case: Ag3Au3
lead1 (p=27, n_ch 729 -> compressed; the system where lsmr landed bad
basins and production pins tr_solver='exact'). Full production
fit_boosted, exact vs gram (1 rep each - walls are minutes):
wall, rms, pole count, PSD min eig, and the fitted model vs the
exact-solver fit on a dense grid. If gram tracks exact here, the
'exact' pin can be reconsidered at a large speedup.
OMP_NUM_THREADS=1.
"""
import gzip
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'Ag3Au3'))

with gzip.open("tmp.gz", "rb") as f:
    R = pickle.load(f)          # pickles first

sys.path.insert(0, ROOT)
from refine_sampling import extract_dense, fit_boosted  # noqa: E402
sys.path.remove(ROOT)

E, F = extract_dense(R, 1, 0)
print(f"[ag3au3] lead1: {len(E)} pts, p={F.shape[1]}", flush=True)

CFG = dict(n_init=24, max_poles=100, target_rms=2e-3,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=5.0, channel_tol=1e-4)

fits = {}
for tag in ('exact', 'gram'):
    kw = dict(CFG)
    kw['tr_solver'] = tag
    t0 = time.time()
    r = fit_boosted(E, F, kw, n_res=8)
    wall = time.time() - t0
    fits[tag] = (r, wall)
    print(f"{tag:6s}: wall {wall:6.1f}s  rms {r.rms:.6g}  n={r.n_poles}",
          flush=True)

r_ex, _ = fits['exact']
r_gr, _ = fits['gram']
Eg = np.linspace(E.min(), E.max(), 801)
F_ex = r_ex.evaluate(Eg)
F_gr = r_gr.evaluate(Eg)
dev = np.max(np.abs(F_gr - F_ex)) / np.max(np.abs(F_ex))
print(f"model max dev gram vs exact (dense grid): {dev:.3e}")
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
sys.path.remove(ROOT)
for tag in ('exact', 'gram'):
    r, wall = fits[tag]
    me = lf.psd_min_eigenvalue(Eg, r.e, r.w, r.C)
    print(f"{tag:6s}: min eig on dense grid {me:.3e}")
print("DONE")
