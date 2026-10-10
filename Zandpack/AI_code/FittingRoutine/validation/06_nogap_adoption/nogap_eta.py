import gzip, pickle, sys, os
import numpy as np

os.chdir('/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/nogap_Ag5')
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'

R = pickle.load(gzip.open('tmp.gz', 'rb'))
RR = pickle.load(gzip.open('tmp.gz', 'rb'))

sys.path.insert(0, ROOT)
from refine_sampling import inject_fits, sampled_transmission_dos  # noqa
sys.path.remove(ROOT)

d = np.load('cmp_fits.npz', allow_pickle=True)
fits = {(l, 0): d[f'lead{l}_ik0'].item() for l in range(2)}

E_ref, T_ref, dos_ref = sampled_transmission_dos(pickle.load(gzip.open('tmp.gz', 'rb')))
print("ref eta used by sampled_transmission_dos: contour imag =", np.asarray(pickle.load(gzip.open('tmp.gz','rb')).Contour)[0].imag)

inject_fits(RR, fits)
from Zandpack.FittingTools import find_correction  # noqa
RR.Renormalise_H(find_correction(RR, Emin=-3.0, Emax=3.0))

Eg = np.linspace(-5, 9, 600)
T_at = np.interp(Eg, E_ref, T_ref)
for eta in [2e-2, 1e-2, 5e-3]:
    _, T_new = RR.Inspect_transmission_from_hilbert_transform(Eg, eta=eta, return_result=True)
    print(f"\neta={eta}:")
    for a, b in [(-5, -3), (-3, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 9)]:
        m = (Eg >= a) & (Eg < b)
        dif = T_new.real[m] - T_at[m]
        print(f"  [{a:>4},{b:>4}): bias {dif.mean():+8.4g}  std {dif.std():8.4g}  rms {np.sqrt((dif**2).mean()):8.4g}")
