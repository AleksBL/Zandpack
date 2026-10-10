"""Realistic non-uniform grid: local density up to ~10x the median around
sharp peaks (the regime the user expects from adaptive refinement).

Same target as nonuniform_test.py but the clusters around sharp poles use
~median/10 spacing instead of ~median/30.
"""
import sys
import time

sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
import numpy as np

from lorentz_fit import fit

rng = np.random.default_rng(3)


def make_target(n_poles=14, p=4):
    e = rng.uniform(-6, 6, n_poles)
    w = 10 ** rng.uniform(-2, 0.3, n_poles)
    C = rng.normal(size=(n_poles, p, p)) + 1j * rng.normal(size=(n_poles, p, p))
    C = np.array([0.5 * (c + c.conj().T) for c in C])
    return e, w, C


def F_true(E, e, w, C):
    out = np.zeros((len(E), C.shape[1], C.shape[2]), dtype=complex)
    for k, Ek in enumerate(E):
        ker = w ** 2 / ((Ek - e) ** 2 + w ** 2)
        out[k] = np.einsum("k,kij->ij", ker, C)
    return out


e, w, C = make_target()

# baseline uniform fill: median spacing 12/40 = 0.3; refine around sharp
# poles at ~10x density (spacing 0.03)
pts = [-6.0]
for ei, wi in zip(e, w):
    if wi < 0.05 and -6 < ei < 6:
        for d in np.linspace(-3 * 0.3, 3 * 0.3, 7):  # +- 0.9 at 0.3 step
            pts.append(ei + d)
        pts += [ei - 3 * wi, ei - wi, ei, ei + wi, ei + 3 * wi]
pts += [6.0]
E_nu = np.unique(np.concatenate([np.array(pts), np.linspace(-6, 6, 40)]))
d = np.diff(E_nu)
print(f"moderate non-uniform grid: N={len(E_nu)}, median dE={np.median(d):.4g}, "
      f"min dE={d.min():.4g} (density ratio {np.median(d)/d.min():.1f}x), "
      f"max dE={d.max():.4g}")

Eg = np.linspace(-8, 8, 801)
Fg = F_true(Eg, e, w, C)
scale = np.sqrt(np.mean(np.abs(Fg) ** 2))

F = F_true(E_nu, e, w, C)
t0 = time.time()
res = fit(E_nu, F, n_init=24, max_poles=60, target_rms=2e-3, psd=True,
          channel_tol=1e-4)
dt = time.time() - t0
Ffit = res.evaluate(Eg)
err = np.sqrt(np.mean(np.abs(Ffit - Fg) ** 2))
print(f"non-unif (moderate): sample rms={res.rms:.4g}  n_poles={len(res.e):3d}  "
      f"dense-grid err={err:.4g} (rel {err / scale:.3g})  [{dt:.0f}s]")
print("widened:", res.history.get("n_widened_subresolution"))
