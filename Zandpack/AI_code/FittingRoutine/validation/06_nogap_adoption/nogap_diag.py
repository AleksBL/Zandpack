import gzip, pickle, sys, os
import numpy as np

os.chdir('/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/nogap_Ag5')
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'

# pickles first
R = pickle.load(gzip.open('tmp.gz', 'rb'))
RR = pickle.load(gzip.open('tmp.gz', 'rb'))

sys.path.insert(0, ROOT)
from refine_sampling import extract_dense, inject_fits, sampled_transmission_dos  # noqa
sys.path.remove(ROOT)

d = np.load('cmp_fits.npz', allow_pickle=True)
fits = {(l, 0): d[f'lead{l}_ik0'].item() for l in range(2)}

# --- Gamma fit quality per band (lead0) ---
E, F = extract_dense(RR, 0, 0)
Ff = fits[(0, 0)].evaluate(E)
err = np.array([np.linalg.norm(Ff[k] - F[k]) for k in range(len(E))])
nrm = np.array([np.linalg.norm(F[k]) for k in range(len(E))])
print("Gamma fit, per band (frobenius rms ||fit-data|| / rms ||data||):")
for a, b in [(-8, -6), (-6, -5), (-5, -4.5), (-4.5, -4), (-4, -3.5), (-3.5, -3), (-3, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 10)]:
    m = (E >= a) & (E < b)
    print(f"  [{a:>5},{b:>5}): rmsres {np.sqrt((err[m]**2).mean()):9.4g}  rmsdata {np.sqrt((nrm[m]**2).mean()):9.4g}")

# --- T curves: bias vs oscillation per band ---
inject_fits(RR, fits)
from Zandpack.FittingTools import find_correction  # noqa
RR.Renormalise_H(find_correction(RR, Emin=-3.0, Emax=3.0))
E_ref, T_ref, dos_ref = sampled_transmission_dos(pickle.load(gzip.open('tmp.gz', 'rb')))
Eg = np.linspace(-5, 9, 600)
_, T_new = RR.Inspect_transmission_from_hilbert_transform(Eg, eta=1e-2, return_result=True)
T_at = np.interp(Eg, E_ref, T_ref)
print("\nT per band: bias (mean T_new-T_ref), oscillation (std of diff), rms, ref range:")
for a, b in [(-5, -3), (-3, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 9)]:
    m = (Eg >= a) & (Eg < b)
    dif = T_new.real[m] - T_at[m]
    print(f"  [{a:>4},{b:>4}): bias {dif.mean():+8.4g}  std {dif.std():8.4g}  rms {np.sqrt((dif**2).mean()):8.4g}"
          f"   [ref T {T_at[m].min():.3g}..{T_at[m].max():.3g}]")

# also with a wider correction window for comparison
RR2 = pickle.load(gzip.open('tmp.gz', 'rb'))
inject_fits(RR2, fits)
RR2.Renormalise_H(find_correction(RR2, Emin=-5.0, Emax=5.0))
_, T_new2 = RR2.Inspect_transmission_from_hilbert_transform(Eg, eta=1e-2, return_result=True)
print("\nwider correction window (-5,5) instead of (-3,3):")
for a, b in [(-5, -3), (-3, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 9)]:
    m = (Eg >= a) & (Eg < b)
    dif = T_new2.real[m] - T_at[m]
    print(f"  [{a:>4},{b:>4}): bias {dif.mean():+8.4g}  std {dif.std():8.4g}  rms {np.sqrt((dif**2).mean()):8.4g}")
