"""The review's synthetic benchmark (lorentz_fit_review.md Appendix A):
p=3 PSD lead, two sqrt band edges (one just outside the window), one
narrow resonance, 161 uniform samples on [-4, 4]. Measures sample rms,
dense-grid between-sample error, and lambda_min off-window."""
import sys
import time

sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
import numpy as np

from lorentz_fit import fit


def g_band(E, c, t):
    x = (E - c) / (2 * t)
    out = np.zeros_like(E)
    m = np.abs(x) < 1
    out[m] = np.sqrt(1 - x[m] ** 2) / t
    return out


def gamma_true(E):
    E = np.asarray(E, float)
    v1 = np.array([1.0, 0.6, 0.2])
    v2 = np.array([0.1, 0.8, -0.5j])
    v3 = np.array([0.0, 0.3, 1.0])
    G = np.zeros((E.size, 3, 3), complex)
    for v, s in ((v1, g_band(E, -1.0, 2.5)), (v2, g_band(E, 1.5, 1.5))):
        G += s[:, None, None] * np.outer(v, v.conj())[None]
    w = 0.03
    G += 0.05 * (w ** 2 / ((E - 0.7) ** 2 + w ** 2))[:, None, None] \
        * np.outer(v3, v3.conj())[None]
    return G


E = np.linspace(-4, 4, 161)
F = gamma_true(E)

Ed = np.linspace(-5, 5, 4001)
Ft = gamma_true(Ed)
inw = np.abs(Ed) <= 4

for tag, kw in (
        ("unc", dict(psd=False)),
        ("psd", dict(psd=True, psd_force=True))):
    t0 = time.time()
    r = fit(E, F, n_init=16, max_poles=30, target_rms=2e-3, **kw)
    Fm = r.evaluate(Ed)
    err = np.linalg.norm(Fm - Ft, axis=(1, 2))
    print(f"{tag}: n_poles={r.n_poles:3d} sample rms={r.rms:.4g} "
          f"dense max in-window={err[inw].max():.4g} "
          f"lam_min={np.linalg.eigvalsh(Fm)[:, 0].min():+.4g} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"   history: n_psd_added={r.history.get('n_psd_added')} "
          f"w_min={r.history.get('w_min'):.4g} "
          f"psd_applied={r.psd['applied'] if r.psd else None}",
          flush=True)

# pole-pair symmetry confirmation (user question): every (e, w) term IS
# the pair z = e +/- i w. Verify numerically: the model evaluated as the
# absorptive part of the conjugate pair equals lorentzian_basis.
r = fit(E, F, n_init=16, max_poles=30, target_rms=2e-3, psd=True,
        psd_force=True)
Eq = np.array([-2.0, 0.0, 1.3])
z = Eq + 0.005j
pair = np.zeros((len(z), r.n_poles), dtype=complex)
for i in range(r.n_poles):
    # absorptive part of C/(z-(e+iw)) + C/(z-(e-iw)) for Hermitian C,
    # evaluated as Im trace with the pair convention
    zc1 = 1.0 / (z - (r.e[i] + 1j * r.w[i]))
    zc2 = 1.0 / (z - (r.e[i] - 1j * r.w[i]))
    pair[:, i] = -0.5j * (zc1 - zc2)  # = w^2/((E-e)^2+w^2) on the axis
from lorentz_fit.varproj import lorentzian_basis
basis = lorentzian_basis(Eq, r.e, r.w)
print("\npole-pair check: |pair-reconstruction - lorentzian_basis| = "
      f"{np.abs(pair.real - basis).max():.3g} (0 => each term is exactly "
      "the conjugate-symmetric pair e +/- i w)")
