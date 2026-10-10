"""Validate the lsmr default: (1) operator branch accepts LinearOperator
jac + explicit tr_solver='lsmr'; (2) fit() default path uses lsmr and
matches the earlier lsmr-tagged RSSE result."""
import lzma
import os
import pickle
import sys
import time

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined'))

def load(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)

R = load('C1.xz')   # pickles first

sys.path.insert(0, ROOT)
import AI_fit  # noqa: E402
from refine_sampling import extract_dense  # noqa: E402
sys.path.remove(ROOT)

# 1. operator branch: force with operator_threshold=0 (LinearOperator jac)
fd = AI_fit.prepare(*extract_dense(load('C1.xz'), 0, 0)[0:2])
E_raw, F_raw = extract_dense(load('C1.xz'), 0, 0)
fd = AI_fit.prepare(E_raw, F_raw)
chans = AI_fit.hermitian_channels(fd.F.shape[1])
B = AI_fit.compress_channels(AI_fit.pack_responses(fd.F, chans))
e0, w0 = AI_fit.linspace_init(fd.E, 24)
bfun = AI_fit.default_bounds(fd.E, w_min=0.01, w_max=0.5)
r = AI_fit.fit_poles(fd.E, B, e0, w0, bounds_fun=bfun, ridge=0.05,
                     operator_threshold=0, tr_solver="lsmr")
print(f"operator branch + lsmr: OK, cost {r['cost']:.3e} nfev {r['nfev']}")

# 2. fit() default path: RSSE production, expect the earlier lsmr numbers
E, F = extract_dense(load('C1.xz'), 0, 0)
CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)
t0 = time.time()
res = AI_fit.fit(E, F, **CFG)
wall = time.time() - t0
print(f"fit() default (expect lsmr): n={res.n_poles} rms={res.rms:.6g} "
      f"wall={wall:.1f}s")
print("earlier lsmr reference: n=42 rms=0.00592872 (standalone fit()); "
      "exact was n=42 rms=0.00592872")
