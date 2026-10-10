import cProfile, pstats, sys, io, time
import numpy as np
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from AI_fit import fit

rng = np.random.default_rng(1)
N, p = 451, 64          # ag3au3-like scale (half p to keep profiling sane)
E = np.linspace(-7, 7, N)
w_t = rng.uniform(0.05, 0.4, 12)
c_t = np.linspace(-6, 6, 12)
Phi = (w_t[None, :] / ((E[:, None] - c_t[None, :])**2 + w_t[None, :]**2)) / np.pi
A = rng.standard_normal((12, p, p)) * 0.1
F = np.einsum("ni,ijk->njk", Phi, A)
F = 0.5 * (F + np.conj(np.swapaxes(F, -1, -2)))
F += 0.05 * np.eye(p)[None]  # background keeps it PSD-ish

t0 = time.time()
pr = cProfile.Profile()
pr.enable()
res = fit(E, F, n_init=24, max_poles=48, target_rms=2e-3,
          psd=True, psd_force=True, eta_raw=1e-2, eta_floor_mult=5.0,
          w_max_raw=0.5, channel_tol=1e-4)
pr.disable()
print(f"FIT DONE n={res.n_poles} rms={res.rms:.4g} in {time.time()-t0:.0f}s")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
print(s.getvalue())
