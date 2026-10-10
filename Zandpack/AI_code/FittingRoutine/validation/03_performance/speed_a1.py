"""Timing: old (backup) AI_fit module vs current AI_fit module on the
identical RSSE production fit. Same data, same config, OMP_NUM_THREADS=1.
"""
import importlib.util
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


# pickles FIRST (shadow rule), then repo imports
R = load('C1.xz')

sys.path.insert(0, ROOT)
import AI_fit as new_mod  # noqa: E402
from refine_sampling import extract_dense  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "AI_fit_old", ROOT + "/backups/AI_fit_2026-10-09_pre_A1_split.py")
old_mod = importlib.util.module_from_spec(spec)
sys.modules["AI_fit_old"] = old_mod   # so numba's cache index resolves
spec.loader.exec_module(old_mod)


CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)

R = load('C1.xz')
E, F = extract_dense(R, 0, 0)
print(f'data: N={len(E)}, p={F.shape[1]}', flush=True)

for name, mod in [("old (backup)", old_mod), ("new (current)", new_mod)]:
    t0 = time.perf_counter()
    r = mod.fit(E, F, **CFG)
    wall = time.perf_counter() - t0
    print(f'{name:14s}: {wall:6.1f}s  n={r.n_poles}  rms={r.rms:.6g}',
          flush=True)
