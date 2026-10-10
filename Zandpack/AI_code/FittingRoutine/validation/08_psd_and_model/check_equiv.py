import sys, time
import numpy as np
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from lorentz_fit.varproj import (hermitian_channels, pack_responses,
                                 unpack_coefficients, channel_indices)
from lorentz_fit.constraints import project_psd, psd_solve_admm

rng = np.random.default_rng(7)
p, M, N = 9, 14, 50
chans = hermitian_channels(p)

# reference (old loop semantics) for pack/unpack
F = (rng.standard_normal((N, p, p)) + 1j * rng.standard_normal((N, p, p)))
F = 0.5 * (F + np.conj(np.swapaxes(F, -1, -2)))
B_ref = np.empty((len(chans), N))
for c, (a, b, part) in enumerate(chans):
    v = F[:, a, b]
    B_ref[c] = v.real if part == "re" else v.imag
B_new = pack_responses(F, chans)
assert np.array_equal(B_ref, B_new), "pack_responses mismatch"

X = rng.standard_normal((M, len(chans)))
C_ref = np.zeros((M, p, p), dtype=complex)
for c, (a, b, part) in enumerate(chans):
    col = X[:, c]
    if part == "re":
        if a == b: C_ref[:, a, a] = col
        else:
            C_ref[:, a, b] += col; C_ref[:, b, a] += col
    else:
        C_ref[:, a, b] += 1j * col; C_ref[:, b, a] -= 1j * col
C_new = unpack_coefficients(X, chans, p)
assert np.allclose(C_ref, C_new, atol=1e-14), "unpack_coefficients mismatch"

# project_psd equivalence
C = C_ref.copy()
P_ref = np.empty_like(C)
for i in range(M):
    lam, V = np.linalg.eigh(0.5 * (C[i] + C[i].conj().T))
    P_ref[i] = (V * np.maximum(lam, 0)) @ V.conj().T
P_new, ncl, worst = project_psd(C)
assert np.allclose(P_ref, P_new, atol=1e-12), "project_psd mismatch"
assert ncl == sum(1 for i in range(M)
                 if np.linalg.eigvalsh(0.5*(C[i]+C[i].conj().T)).min() < 0)
print("EQUIVALENCE PASS")
