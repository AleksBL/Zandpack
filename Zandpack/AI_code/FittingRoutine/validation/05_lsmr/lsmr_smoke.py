"""Step 5 smoke test: fit_poles with tr_solver='lsmr' vs default 'exact'
on RSSE-shaped problems (cold skrf init). Records wall, nfev, cost."""
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
import AI_fit as lf  # noqa: E402
from refine_sampling import extract_dense  # noqa: E402
sys.path.remove(ROOT)

E_raw, F_raw = extract_dense(R, 0, 0)
fd = lf.prepare(E_raw, F_raw)
chans = lf.hermitian_channels(fd.F.shape[1])
B = lf.compress_channels(lf.pack_responses(fd.F, chans))

for M in [24, 50, 100]:
    e0, w0 = lf.linspace_init(fd.E, M)
    bfun = lf.default_bounds(fd.E, w_min=0.005, w_max=0.5)
    lo, hi = bfun(M)
    e0 = np.clip(e0, lo[0::2], hi[0::2])
    w0 = np.clip(w0, lo[1::2], hi[1::2])
    out = {}
    for tag, ts in [("exact", None), ("lsmr", "lsmr")]:
        t0 = time.perf_counter()
        r = lf.fit_poles(fd.E, B, e0, w0, bounds_fun=bfun,
                         ridge=0.05, tr_solver=ts)
        out[tag] = (time.perf_counter() - t0, r["cost"], r["nfev"])
        print(f"M={M:3d} {tag:5s}: wall {out[tag][0]:6.2f}s  "
              f"cost {r['cost']:.6e}  nfev {r['nfev']}", flush=True)
    print(f"  -> lsmr/exact wall: {out['lsmr'][0] / out['exact'][0]:.2f}  "
          f"cost ratio: {out['lsmr'][1] / max(out['exact'][1], 1e-300):.4f}")
