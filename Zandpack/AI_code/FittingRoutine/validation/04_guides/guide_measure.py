"""Guide step 1: MEASURE on the RSSE production fit (lead0):
- _varproj_factors call count, dense-jac call count (residual_and_jacobian)
- cost-only vs jacobian-bearing calls (fun at trial points = the A1 target)
- SVD-fallback frequency (np.linalg.lstsq count from _varproj_factors)
"""
import gzip
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined'))

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f) if path.endswith('.pkl') else None

with open('C1.xz', 'rb') as f:
    pass

def load_xz(path):
    import lzma
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)

R = load_xz('C1.xz')   # pickles first

sys.path.insert(0, ROOT)
import AI_fit  # noqa: E402
from refine_sampling import extract_dense, fit_boosted  # noqa: E402

CNT = defaultdict(int)

_vp_orig = AI_fit._varproj_factors
_rj_orig = AI_fit.residual_and_jacobian
_ls_orig = np.linalg.lstsq
_qr_orig = np.linalg.qr

def _vp(*a, **k):
    CNT['varproj_calls'] += 1
    return _vp_orig(*a, **k)

def _rj(*a, **k):
    CNT['dense_jac_calls'] += 1
    return _rj_orig(*a, **k)

def _ls(*a, **k):
    CNT['lstsq_calls'] += 1
    return _ls_orig(*a, **k)

AI_fit._varproj_factors = _vp
AI_fit.residual_and_jacobian = _rj
np.linalg.lstsq = _ls

E, F = extract_dense(R, 0, 0)
CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)
t0 = time.time()
r = fit_boosted(E, F, dict(CFG), n_res=8)
wall = time.time() - t0

# restore (this process ends anyway)
vp, jac, cost_only = (CNT['varproj_calls'], CNT['dense_jac_calls'],
                      CNT['varproj_calls'] - CNT['dense_jac_calls'])
print(f"\nfit: {r.n_poles} poles, rms {r.rms:.4g}, {wall:.1f}s")
print(f"_varproj_factors calls : {vp}")
print(f"dense-jac calls        : {jac}   (each needs Z/K/S)")
print(f"cost-only calls        : {cost_only}  "
      f"({100 * cost_only / max(vp, 1):.1f}% of all -> A1 saving target)")
print(f"np.linalg.lstsq calls  : {CNT['lstsq_calls']} "
      f"(fallback fires ~count/3 times if only _varproj uses it)")
