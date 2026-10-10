"""Relaxed gate on the big-J system: nogap lead0 fit with
tr_solver='lsmr' vs the exact-mode reference from the memory run
(wall 659.3s, n=62, rms=0.00176817). One lsmr run; quality compared
on rms/n_poles/Gamma-vs-data."""
import gzip
import os
import pickle
import sys
import time

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'nogap_Ag5'))

def load(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

R = load('tmp.gz')   # pickles first

sys.path.insert(0, ROOT)
from refine_sampling import extract_dense  # noqa: E402
sys.path.remove(ROOT)
import AI_fit  # noqa: E402

E, F = extract_dense(R, 0, 0)
CFG = dict(n_init=24, max_poles=62, target_rms=2e-3,
           psd=True, psd_force=False, eta_raw=2e-2,
           local_floor_frac=1.5, w_max_raw=0.3,
           eta_floor_mult=1.0, channel_tol=1e-4,
           tr_solver='lsmr')
t0 = time.time()
r = AI_fit.fit(E, F, **CFG)
wall = time.time() - t0
print(f"nogap lead0 LSMR: n={r.n_poles} rms={r.rms:.6g} "
      f"wall={wall:.1f}s", flush=True)
print("reference EXACT (mem run): n=62 rms=0.00176817 wall=659.3s")
