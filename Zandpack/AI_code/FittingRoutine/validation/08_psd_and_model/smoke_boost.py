import sys, numpy as np
ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, ROOT)
from lorentz_fit import fit as lfit
from lorentz_fit.validate import frobenius_rms_error
from refine_sampling import fit_boosted

E = np.linspace(-5, 5, 120)
e_true = np.array([-2.0, 0.3, 1.5]); w_true = np.array([0.05, 0.2, 0.4])
Phi = 1.0/np.pi * (w_true[None,:]/((E[:,None]-e_true[None,:])**2 + w_true[None,:]**2))
F = np.zeros((E.size, 2, 2))
F[:,0,0] = Phi @ np.array([1.0, 2.0, 0.5])
F[:,1,1] = Phi @ np.array([0.7, 1.5, 0.9])
F[:,0,1] = Phi @ np.array([0.3, 0.4, 0.2])
F[:,1,0] = F[:,0,1]

base_kw = dict(n_init=24, max_poles=60, psd=True, psd_force=True,
               eta_raw=1e-2, eta_floor_mult=5.0, w_max_raw=0.5,
               local_floor_frac=1.5, target_rms=2e-3)
r1 = lfit(E, F, **base_kw)
rel1 = frobenius_rms_error(r1.evaluate(E), F)
print("base:", r1.n_poles, "rel", rel1)
comb = fit_boosted(E, F, base_kw, n_res=8)
relc = frobenius_rms_error(comb.evaluate(E), F)
print("comb:", comb.n_poles, "rel", relc, "parts", comb.history["boost_parts"])
Fd = comb.evaluate(np.linspace(-7,7,501))
print("minEig", np.linalg.eigvalsh(Fd).min())
assert comb.history["boost_parts"][1] <= 10, "residual fit grew!"
print("improved:", relc <= rel1)
print("SMOKE_OK")
