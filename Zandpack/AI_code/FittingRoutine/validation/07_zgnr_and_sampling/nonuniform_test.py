"""Synthetic check: does lorentz_fit work on strongly non-uniform grids?

Builds a random shared-pole Lorentzian matrix function, samples it on a
strongly non-uniform grid (dense near sharp poles, sparse in flat
regions), fits, and compares against the truth on a dense uniform grid.
Also compares against a uniform grid with the same number of points.
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

# --- strongly non-uniform grid -------------------------------------------
pts = [-6.0]
for ei, wi in zip(e, w):
    if wi < 0.05 and -6 < ei < 6:
        pts += [ei - 3 * wi, ei - wi, ei, ei + wi, ei + 3 * wi]
pts += [6.0]
E_nu = np.unique(np.concatenate([np.array(pts), np.linspace(-6, 6, 40)]))
print(f"non-uniform grid: N={len(E_nu)}, "
      f"min dE={np.diff(E_nu).min():.4g}, max dE={np.diff(E_nu).max():.4g}")

# --- uniform grid with the same N ----------------------------------------
E_u = np.linspace(-6, 6, len(E_nu))

Eg = np.linspace(-8, 8, 801)
Fg = F_true(Eg, e, w, C)
scale = np.sqrt(np.mean(np.abs(Fg) ** 2))

for tag, E in (("uniform  ", E_u), ("non-unif", E_nu)):
    F = F_true(E, e, w, C)
    t0 = time.time()
    res = fit(E, F, n_init=24, max_poles=60, target_rms=2e-3, psd=True,
              channel_tol=1e-4)
    dt = time.time() - t0
    Ffit = res.evaluate(Eg)
    err = np.sqrt(np.mean(np.abs(Ffit - Fg) ** 2))
    print(f"{tag}: sample rms={res.rms:.4g}  n_poles={len(res.e):3d}  "
          f"dense-grid err={err:.4g} (rel {err / scale:.3g})  [{dt:.0f}s]")
