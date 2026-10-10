"""Validate the equal-pole-count requirement (multi-k systems):
1. pad_poles / equalize_pole_counts: inertness + count equalization.
2. inject_fits(equal_nl='all') (new default): every (lead, ik)
   container carries exactly R.NumL poles.
3. Physics unchanged vs the legacy 'lead' injection (same fits).
Real system: zgnr_spin (2 leads x 2 k-points), exact-era cached fits.
"""
import pickle
import sys

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
import os
os.chdir(ROOT + '/Data2/zgnr_spin')

# pickles first
with open('tmp.gz', 'rb') as f:
    pass
import gzip
with gzip.open('tmp.gz', 'rb') as f:
    R_all = pickle.load(f)
with gzip.open('tmp.gz', 'rb') as f:
    R_lead = pickle.load(f)

sys.path.insert(0, ROOT)
import AI_fit  # noqa: E402
from refine_sampling import inject_fits, fit_boosted  # noqa: E402
sys.path.remove(ROOT)
from Zandpack.FittingTools import find_correction  # noqa: E402

d = np.load('refined/cmp_fits_exactera.npz', allow_pickle=True)
fits = {}
for l in range(2):
    for ik in range(2):
        fits[(l, ik)] = d[f'lead{l}_ik{ik}'].item()
print("cached fit pole counts:",
      {k: v.n_poles for k, v in fits.items()})

# --- 1. pad_poles / equalize_pole_counts ---
Eg = np.linspace(-12, 12, 101)
eq = AI_fit.equalize_pole_counts(fits)
print("equalized counts:", {k: v.n_poles for k, v in eq.items()})
ok = True
for k, f in fits.items():
    same = np.array_equal(f.evaluate(Eg), eq[k].evaluate(Eg))
    ok &= same
    if not same:
        print(f"  MISMATCH {k}")
print("evaluate identity after padding:", "PASS" if ok else "FAIL")
eq0 = AI_fit.equalize_pole_counts(fits, mode="zero")
ok2 = all(np.array_equal(fits[k].evaluate(Eg), eq0[k].evaluate(Eg))
          for k in fits)
print("zero-mode evaluate identity:", "PASS" if ok2 else "FAIL")

# --- 2/3. inject with 'all' (new default) vs 'lead' (legacy) ---
inject_fits(R_all, fits)                # default equal_nl='all'
inject_fits(R_lead, fits, equal_nl='lead')
NL_all = max(f.n_poles for f in fits.values())
print(f"\nR.NumL: all-mode {R_all.NumL}, lead-mode {R_lead.NumL}, "
      f"expected max {NL_all}")
shapes_ok = True
for lead in range(2):
    ei_a = np.asarray(R_all.NO_fitted_lorentzians[lead].ei)
    ei_l = np.asarray(R_lead.NO_fitted_lorentzians[lead].ei)
    print(f"  lead{lead}: ei shape all={ei_a.shape} lead={ei_l.shape}")
    shapes_ok &= ei_a.shape[1] == NL_all
print("all-mode shapes == (n_ik, NL_all) for every lead:",
      "PASS" if shapes_ok else "FAIL")

R_all.Renormalise_H(find_correction(R_all, Emin=-2, Emax=2.0))
R_lead.Renormalise_H(find_correction(R_lead, Emin=-2, Emax=2.0))
Eg2 = np.linspace(-7, 7, 300)
_, T_a = R_all.Inspect_transmission_from_hilbert_transform(
    Eg2, eta=1e-2, return_result=True)
_, T_l = R_lead.Inspect_transmission_from_hilbert_transform(
    Eg2, eta=1e-2, return_result=True)
dif = np.max(np.abs(T_a.real - T_l.real))
print(f"\nT max|all - lead| = {dif:.3e}  "
      f"(T range {T_l.real.min():.3g}..{T_l.real.max():.3g})")
print("physics unchanged:", "PASS" if dif < 1e-8 else "CHECK")
