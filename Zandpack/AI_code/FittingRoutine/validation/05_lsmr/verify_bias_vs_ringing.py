"""Verify the lsmr_vs_exact_edge_band.md claim: is the lsmr [3,5] error
a smooth BIAS (same-sign offset) rather than RINGING (oscillation)?
Applies the document's section-6 split to both solvers' residuals."""
import lzma
import os
import pickle
import sys

import numpy as np
from scipy.signal import savgol_filter

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

curves = {}
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
    Eg = np.linspace(-5, 5, 2000)
    _, T_new = RR.Inspect_transmission_from_hilbert_transform(
        Eg, eta=1e-2, return_result=True)
    curves[tag] = T_new.real
    print(f"[{tag}] done", flush=True)

Eg = np.linspace(-5, 5, 2000)
T_at = np.interp(Eg, E_ref, T_ref)


def split_residual(E, T_fit, T_ref, lo=3.0, hi=5.0,
                   smooth_width_eV=0.5, polyorder=3):
    m = (E >= lo) & (E <= hi)
    Ew, r = E[m], (T_fit - T_ref)[m]
    dE = np.median(np.diff(Ew))
    win = int(round(smooth_width_eV / dE))
    win += (win % 2 == 0)
    win = min(win, len(r) if len(r) % 2 else len(r) - 1)
    smooth = savgol_filter(r, win, polyorder)
    ripple = r - smooth
    integ = lambda y: np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(Ew))
    return dict(bias_mean=smooth.mean(),
                bias_max=np.abs(smooth).max(),
                ringing_rms=np.sqrt(np.mean(ripple ** 2)),
                rms_total=float(np.sqrt(np.mean(r ** 2))),
                int_rel_err=integ(T_fit[m] - T_ref[m]) / integ(T_ref[m]))


print(f"\n{'solver':8s} {'bias_mean':>10s} {'bias_max':>9s} "
      f"{'ringing_rms':>11s} {'rms_total':>9s} {'int_rel_err':>11s}")
for tag in ['exact', 'lsmr']:
    s = split_residual(Eg, curves[tag], T_at)
    print(f"{tag:8s} {s['bias_mean']:+10.4f} {s['bias_max']:9.4f} "
          f"{s['ringing_rms']:11.4f} {s['rms_total']:9.4f} "
          f"{s['int_rel_err']:+11.4f}")
print("\nzero-sign test: same-sign fraction of the residual")
for tag in ['exact', 'lsmr']:
    m = (Eg >= 3.0) & (Eg <= 5.0)
    r = (curves[tag] - T_at)[m]
    print(f"  {tag}: {np.mean(np.sign(r) == np.sign(np.median(r))):.2f}")
