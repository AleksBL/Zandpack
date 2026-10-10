"""End-to-end timing on nogap_Ag5 lead0 (dense-J path, N=1001):
old (pre-jac-kernel backup) vs new (numba kernel) fit_boosted,
production config. Bitwise result comparison."""
import gzip
import importlib.util
import os
import pickle
import sys
import time

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'nogap_Ag5'))

# pickles first
def load(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

R = load('tmp.gz')

sys.path.insert(0, ROOT)
import AI_fit as new  # noqa: E402
from refine_sampling import extract_dense, fit_boosted  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "AI_fit_old", ROOT + "/backups/AI_fit_2026-10-09_pre_jac_kernel.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

CFG = dict(n_init=24, max_poles=62, target_rms=2e-3,
           psd=True, psd_force=False, eta_raw=2e-2,
           local_floor_frac=1.5, w_max_raw=0.3,
           eta_floor_mult=1.0,
           channel_tol=1e-4)

E, F = extract_dense(R, 0, 0)
print(f'data: N={len(E)}, p={F.shape[1]}', flush=True)

for tag, fb in [('old (backup)', fit_boosted), ]:
    pass  # fit_boosted lives in refine_sampling (new); build old path below

# old fit_boosted via the backup module's own functions
def old_fit_boosted(E, F, base_kw, n_res=8):
    res_kw = dict(base_kw or {})
    res_kw['n_poles'] = int(n_res)
    res_kw['target_rms'] = 0.0
    res_kw.pop('max_poles', None)
    res_kw.pop('init_poles', None)
    res_kw['psd'] = True
    res_kw['psd_force'] = True
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    r1 = old.fit(E, F, **base_kw)
    resid = F - r1.evaluate(E)
    r2 = old.fit(E, resid, **res_kw)
    comb = refine_cls([r1, r2])
    comb.rms = float(frobenius(comb.evaluate(E), F))
    return comb

# CombinedFit + frobenius from the backup module
refine_cls = old.CombinedFit
frobenius = old.frobenius_rms_error

results = {}
t0 = time.time()
ro = old_fit_boosted(E, F, dict(CFG), n_res=8)
results['old'] = (time.time() - t0, ro)
print(f"old (backup): {results['old'][0]:.1f}s  n={ro.n_poles} "
      f"rms={ro.rms:.6g}", flush=True)

t0 = time.time()
rn = fit_boosted(E, F, dict(CFG), n_res=8)
results['new'] = (time.time() - t0, rn)
print(f"new (current): {results['new'][0]:.1f}s  n={rn.n_poles} "
      f"rms={rn.rms:.6g}", flush=True)

same = np.array_equal(ro.e, rn.e) and np.array_equal(ro.w, rn.w) \
    and np.array_equal(ro.C, rn.C)
print(f"bitwise identical: {same}")
print(f"speedup: {results['old'][0] / results['new'][0]:.2f}x")
