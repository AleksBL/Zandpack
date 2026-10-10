"""Step 5 relaxed gate (speedupguide_new.md section 8): full RSSE
production fit, tr_solver='lsmr' vs default 'exact'.
3 alternating A/B runs (medians); quality: final rms, pole count, PSD
min eig, and the fitted Gamma compared between the two best fits."""
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

R = load('C1.xz')   # pickles first

sys.path.insert(0, ROOT)
from refine_sampling import extract_dense, fit_boosted  # noqa: E402
sys.path.remove(ROOT)

CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)

E, F = extract_dense(R, 0, 0)
runs = {'exact': [], 'lsmr': []}
fits = {}
for rep in range(3):
    for tag in ['exact', 'lsmr']:
        kw = dict(CFG)
        if tag == 'lsmr':
            kw['tr_solver'] = 'lsmr'
        t0 = time.time()
        r = fit_boosted(E, F, kw, n_res=8)
        wall = time.time() - t0
        runs[tag].append((wall, r.rms, r.n_poles))
        fits[tag] = r
        print(f"rep{rep} {tag:5s}: wall {wall:5.1f}s  rms {r.rms:.6g}  "
              f"n={r.n_poles}", flush=True)

for tag in ['exact', 'lsmr']:
    walls = sorted(w[0] for w in runs[tag])
    print(f"{tag:5s}: median wall {walls[1]:.1f}s  "
          f"rms {runs[tag][-1][1]:.6g}  n {runs[tag][-1][2]}")

# quality comparison of the two final fits
Eg = np.linspace(-6, 6, 301)
Fe = fits['exact'].evaluate(Eg)
Fl = fits['lsmr'].evaluate(Eg)
scale = np.abs(Fe).max()
print(f"Gamma comparison: max|exact-lsmr| = {np.abs(Fe - Fl).max():.3e} "
      f"(scale {scale:.3e}, rel {np.abs(Fe - Fl).max() / scale:.3e})")
for tag in ['exact', 'lsmr']:
    me = float(np.linalg.eigvalsh(fits[tag].evaluate(Eg)).min())
    print(f"{tag:5s}: minEig on grid {me:+.3e}")
