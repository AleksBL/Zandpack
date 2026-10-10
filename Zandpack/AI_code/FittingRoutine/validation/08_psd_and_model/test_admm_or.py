"""Correctness of the over-relaxed ADMM (guide 2.3): the solution must
be PSD and match the tightly-converged reference objective."""
import sys
import numpy as np
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from lorentz_fit.varproj import hermitian_channels, unpack_coefficients
from lorentz_fit.constraints import psd_solve_admm

rng = np.random.default_rng(3)
p, M, N = 8, 12, 60
chans = hermitian_channels(p)
E = np.linspace(-1, 1, N)
e0 = np.linspace(-0.8, 0.8, M)
w0 = rng.uniform(0.03, 0.2, M)
Phi = (w0[None, :] / ((E[:, None] - e0[None, :]) ** 2
                      + w0[None, :] ** 2)) / np.pi
A0 = rng.standard_normal((M, p, p))
Y = np.einsum("ni,ijk->njk", Phi, A0)
B = np.stack([Y[:, a, b].real if part == "re" else Y[:, a, b].imag
              for a, b, part in chans])  # (n_ch, N)

def objective(Z):
    Fz = np.einsum("ni,ijk->njk", Phi, Z)
    return float(np.sum(np.abs(Fz - Y) ** 2))

Z_new, _ = psd_solve_admm(Phi, B, chans, p, iters=200)
Z_ref, _ = psd_solve_admm(Phi, B, chans, p, iters=5000)
o_new, o_ref = objective(Z_new), objective(Z_ref)
psd_min = float(np.linalg.eigvalsh(Z_new).min())
print(f"objective new(200 its): {o_new:.6e}")
print(f"objective ref(5000 its): {o_ref:.6e}")
print(f"relative gap: {abs(o_new-o_ref)/max(o_ref,1e-300):.3e}")
print(f"min eigenvalue of solution: {psd_min:+.3e}")
assert psd_min >= -1e-12, "solution not PSD!"
assert abs(o_new - o_ref) <= 1e-4 * max(o_ref, 1e-300), \
    "over-relaxed ADMM did not reach the reference optimum"
print("ADMM_OVERRELAX_PASS")
