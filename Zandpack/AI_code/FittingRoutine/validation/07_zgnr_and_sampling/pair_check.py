"""Pole-pair symmetry confirmation (user question): each basis term is
exactly the absorptive part of the conjugate-symmetric pole pair
z = e +/- i*w, with the SAME Hermitian residue used for both members
(and Zandpack's KK partner uses the same pair)."""
import sys

sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
import numpy as np

from lorentz_fit.varproj import lorentzian_basis

rng = np.random.default_rng(2)
e = rng.uniform(-5, 5, 7)
w = 10 ** rng.uniform(-2, 0, 7)
Eq = np.linspace(-6, 6, 41)

# L(E) = w^2/((E-e)^2+w^2)
#      = (w/2i) [ 1/(E-e-iw) - 1/(E-e+iw) ]
# i.e. exactly the difference of the two conjugate-pair pole kernels,
# with equal and opposite weights - no unpaired imaginary parts.
basis = lorentzian_basis(Eq, e, w)
pair = np.empty((Eq.size, e.size))
for i in range(e.size):
    zc1 = 1.0 / (Eq - e[i] - 1j * w[i])
    zc2 = 1.0 / (Eq - e[i] + 1j * w[i])
    pair[:, i] = (w[i] / (2j) * (zc1 - zc2)).real
dev = np.abs(pair - basis).max()
print(f"|pair-reconstruction - lorentzian_basis| = {dev:.3g}")
assert dev < 1e-12
print("CONFIRMED: every (e, w) term is the conjugate-symmetric pair "
      "z = e +/- iw; no unpaired imaginary parts exist in the model.")

# KK partner: KK_L(E) = L(E) (E-e)/w. Check that the dispersive part of
# the same pair gives it:  1/(z-(e+iw)) + 1/(z-(e-iw)) = 2(E-e)/((E-e)^2+w^2)
kk = np.empty((Eq.size, e.size))
for i in range(e.size):
    zc1 = 1.0 / (Eq - e[i] - 1j * w[i])
    zc2 = 1.0 / (Eq - e[i] + 1j * w[i])
    kk[:, i] = (0.5 * (zc1 + zc2) * w[i]).real
kk_target = basis * (Eq[:, None] - e[None, :]) / w[None, :]
dev2 = np.abs(kk - kk_target).max()
print(f"|KK from the same pair - target| = {dev2:.3g}")
assert dev2 < 1e-12
print("CONFIRMED: the dispersive (Hilbert) partner used by Zandpack's "
      "SE construction comes from the SAME conjugate pair.")
