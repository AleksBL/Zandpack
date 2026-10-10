"""Synthetic PSD test: does the PSD-aware adaptive loop reduce the
projected model's rms vs the old project-at-the-end behavior?"""
import sys
import time

sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
import numpy as np

from lorentz_fit import fit

rng = np.random.default_rng(11)


def make(n_poles=16, p=5):
    e = rng.uniform(-6, 6, n_poles)
    w = 10 ** rng.uniform(-1.5, 0.5, n_poles)
    C = np.zeros((n_poles, p, p), dtype=complex)
    for i in range(n_poles):
        A = rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))
        C[i] = A @ A.conj().T          # PSD residues by construction
    return e, w, C


def F_of(E, e, w, C):
    out = np.zeros((len(E), C.shape[1], C.shape[2]), dtype=complex)
    for k, Ek in enumerate(E):
        ker = w ** 2 / ((Ek - e) ** 2 + w ** 2)
        out[k] = np.einsum("k,kij->ij", ker, C)
    return out


e, w, C = make()
E = np.linspace(-6, 6, 120)
F = F_of(E, e, w, C)

t0 = time.time()
res = fit(E, F, n_init=24, max_poles=100, target_rms=2e-3, psd=True,
          psd_force=True)
print(f"psd fit: rms={res.rms:.4g} n_poles={res.n_poles} "
      f"psd_applied={res.psd['applied']} "
      f"n_psd_added={res.history.get('n_psd_added')} "
      f"[{time.time()-t0:.0f}s]")
# PSD check on dense grid
Eg = np.linspace(-8, 8, 401)
me = np.linalg.eigvalsh(res.evaluate(Eg)).min(axis=1)
print(f"min eig of total fit on dense grid: {me.min():+.3g} (must be >= 0)")
# accuracy vs truth
Fg = F_of(Eg, e, w, C)
err = np.sqrt(np.mean(np.abs(res.evaluate(Eg) - Fg) ** 2))
print(f"dense-grid truth error: {err:.4g} (scale "
      f"{np.sqrt(np.mean(np.abs(Fg)**2)):.4g})")
