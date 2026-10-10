"""never_form_J guide, validation steps 6-8: full-fit trajectory
comparison + timing, real VerySmallRSSE data.

A) fit_poles trajectory (relaxed gate: nfev, final cost, fitted poles)
   exact vs lsmr vs gram, both leads, n=59 cold start.
B) production fit_boosted (full adaptive loop + PSD + boost) wall time
   and quality, alternating A/B/C x 3 reps, medians.
OMP_NUM_THREADS=1.
"""
import lzma
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined'))


def load(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)


R = load('C1.xz')   # pickles first

sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
from refine_sampling import extract_dense, fit_boosted  # noqa: E402
sys.path.remove(ROOT)

SOLVERS = ['exact', 'lsmr', 'gram']

print("=== A) fit_poles trajectory (cold linspace n=59, max_nfev=400) ===")
for lead in (0, 1):
    E_raw, F_raw = extract_dense(R, lead, 0)
    fd = lf.prepare(E_raw, F_raw)
    chans = lf.hermitian_channels(fd.F.shape[1])
    B = lf.pack_responses(fd.F, chans)
    sw = lf._sample_weights(fd.F, "relative", quad=lf._quad_widths(fd.E))
    e0, w0 = lf.linspace_init(fd.E, 59)
    res = {}
    for tag in SOLVERS:
        t0 = time.time()
        r = lf.fit_poles(fd.E, B, e0, w0, sample_weights=sw, ridge=0.05,
                         bounds_fun=lf.default_bounds(fd.E),
                         max_nfev=400, tr_solver=tag)
        r['wall'] = time.time() - t0
        res[tag] = r
        print(f"lead{lead} {tag:6s} cost {r['cost']:.8e} nfev {r['nfev']:4d} "
              f"wall {r['wall']:6.2f}s  e_maxdiff_vs_exact "
              f"{np.max(np.abs(r['e'] - res['exact']['e'])):.2e}  "
              f"w_maxdiff {np.max(np.abs(r['w'] - res['exact']['w'])):.2e}",
              flush=True)

print("\n=== B) production fit_boosted, 3 alternating reps ===")
CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)

E, F = extract_dense(R, 0, 0)
runs = {t: [] for t in SOLVERS}
fits = {}
for rep in range(3):
    for tag in SOLVERS:
        kw = dict(CFG)
        kw['tr_solver'] = tag
        t0 = time.time()
        r = fit_boosted(E, F, kw, n_res=8)
        wall = time.time() - t0
        runs[tag].append((wall, r.rms, r.n_poles))
        fits[tag] = r
        print(f"rep{rep} {tag:6s}: wall {wall:5.1f}s  rms {r.rms:.6g}  "
              f"n={r.n_poles}", flush=True)

print()
for tag in SOLVERS:
    walls = sorted(w[0] for w in runs[tag])
    print(f"{tag:6s}: median wall {walls[1]:6.1f}s  "
          f"rms {runs[tag][-1][1]:.6g}  n {runs[tag][-1][2]}")

# model comparison between the three final fits (dense grid, lead0)
Eg = np.linspace(E.min(), E.max(), 1201)
for tag in SOLVERS:
    r = fits[tag]
    Fg = r.evaluate(Eg)
    if tag == 'exact':
        Fref = Fg
        continue
    dev = np.max(np.abs(Fg - Fref)) / np.max(np.abs(Fref))
    print(f"model max dev {tag} vs exact: {dev:.2e}")
print("DONE")
