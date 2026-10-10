"""Diagnose the non-uniform-grid failure: where and why does the fit go wrong?"""
import sys

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
pts = [-6.0]
for ei, wi in zip(e, w):
    if wi < 0.05 and -6 < ei < 6:
        pts += [ei - 3 * wi, ei - wi, ei, ei + wi, ei + 3 * wi]
pts += [6.0]
E_nu = np.unique(np.concatenate([np.array(pts), np.linspace(-6, 6, 40)]))

F = F_true(E_nu, e, w, C)
res = fit(E_nu, F, n_init=24, max_poles=60, target_rms=2e-3, psd=True,
          channel_tol=1e-4)

Eg = np.linspace(-8, 8, 4001)
Fg = F_true(Eg, e, w, C)
Ffit = res.evaluate(Eg)
err = np.sqrt(np.mean(np.abs(Ffit - Fg) ** 2, axis=(1, 2)))

print("history:", {k: v for k, v in res.history.items()
                   if not isinstance(v, (list, dict))})
print(f"w_min used (scaled): {res.history.get('w_min')}")
print()
# where is the error concentrated?
bad = np.where(err > 10 * np.median(err))[0]
if bad.size:
    print(f"error hotspots: E in [{Eg[bad[0]]:.2f}, {Eg[bad[-1]]:.2f}], "
          f"max err {err.max():.3g} at E={Eg[np.argmax(err)]:.2f}")
# error vs distance to nearest sample
d = np.min(np.abs(Eg[:, None] - E_nu[None, :]), axis=1)
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.16)):
    m = (d >= lo) & (d < hi)
    if m.any():
        print(f"dist-to-nearest-sample [{lo},{hi}): "
              f"mean err {err[m].mean():.3g}  max {err[m].max():.3g}")
print()
# poles vs local spacing
Es = np.sort(E_nu)
def local_spacing(x):
    j = np.searchsorted(Es, x)
    j = min(max(j, 1), len(Es) - 1)
    return min(x - Es[j - 1], Es[j] - x) if 0 < j < len(Es) else np.inf
print("poles (raw units) vs local half-spacing:")
for ei, wi in sorted(zip(res.e, res.w)):
    ls = local_spacing(ei)
    flag = "  <== NARROWER THAN LOCAL SPACING" if wi < 0.5 * ls else ""
    print(f"  e={ei:+7.3f}  w={wi:8.4f}  local dE={ls:.4f}{flag}")
print()
print("true poles:")
for ei, wi in sorted(zip(e, w)):
    print(f"  e={ei:+7.3f}  w={wi:8.4f}")
