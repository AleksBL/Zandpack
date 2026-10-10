import sys, numpy as np
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from lorentz_fit import fit
E = np.linspace(-5, 5, 120)
e_t = np.array([-2.0, 0.3, 1.5]); w_t = np.array([0.05, 0.2, 0.4])
Phi = 1.0/np.pi * (w_t[None,:]/((E[:,None]-e_t[None,:])**2 + w_t[None,:]**2))
F = np.zeros((E.size, 2, 2))
F[:,0,0] = Phi @ np.array([1.0, 2.0, 0.5])
F[:,1,1] = Phi @ np.array([0.7, 1.5, 0.9])
F[:,0,1] = Phi @ np.array([0.3, 0.4, 0.2]); F[:,1,0] = F[:,0,1]
res = fit(E, F, n_init=24, max_poles=60, target_rms=2e-3,
          psd=True, psd_force=True, eta_raw=1e-2, eta_floor_mult=5.0,
          w_max_raw=0.5, verbose=True)
print("RESULT:", res.n_poles, res.rms)
