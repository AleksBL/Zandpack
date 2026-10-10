"""Relaxed gate, T level: RSSE exact-fit vs lsmr-fit, injected and
compared at the transmission level (band rms vs sampled-SE reference)."""
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

R0 = load('C1.xz')   # pickles first

sys.path.insert(0, ROOT)
from refine_sampling import (extract_dense, fit_boosted, inject_fits,  # noqa
                             sampled_transmission_dos)
sys.path.remove(ROOT)
from Zandpack.FittingTools import find_correction  # noqa: E402

CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)

E_ref, T_ref, dos_ref = sampled_transmission_dos(load('C1.xz'))

results = {}
for tag in ['exact', 'lsmr']:
    kw = dict(CFG)
    if tag == 'lsmr':
        kw['tr_solver'] = 'lsmr'
    fits = {}
    for lead in range(2):
        Rl = load('C1.xz')
        E, F = extract_dense(Rl, lead, 0)
        fits[(lead, 0)] = fit_boosted(E, F, kw, n_res=8)
    RR = load('C1.xz')
    inject_fits(RR, fits)
    RR.Renormalise_H(find_correction(RR))
    Eg = np.linspace(-5, 5, 1000)
    _, T_new = RR.Inspect_transmission_from_hilbert_transform(
        Eg, eta=1e-2, return_result=True)
    results[tag] = T_new.real
    print(f"{tag}: fits done", flush=True)

T_at = np.interp(Eg, E_ref, T_ref)


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


for tag in ['exact', 'lsmr']:
    print(f"\n{tag}: overall T rms {rms(results[tag], T_at):.4g}")
    for a, b in [(-5, -3), (-3, -1), (-1, 0.3), (0.3, 1), (1, 3), (3, 5)]:
        m = (Eg >= a) & (Eg < b)
        print(f"  [{a:>4},{b:>4}): rms {rms(results[tag][m], T_at[m]):8.4g}")
    i0 = int(np.argmin(np.abs(Eg)))
    print(f"  T(0) fit {results[tag][i0]:.4f} (ref {T_at[i0]:.4f})")
