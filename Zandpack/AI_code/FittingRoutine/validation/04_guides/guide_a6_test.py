"""Guide A6: unpadded ridge products — skip the zero ridge rows in the
big products (Q^T@D, Q@(...), Q^T@Y, A@X). QR itself stays on the
padded A. Mathematically exact; bitwise NOT guaranteed (different GEMM
shapes) -> the exact gate decides adoption.
"""
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402


def factors_core_a6(E, B, theta, sw, rcond, ridge):
    """Core with unpadded products: X, Res computed from the top blocks."""
    e, w = theta[0::2], theta[1::2]
    Phi = lf.lorentzian_basis(E, e, w)
    A, Y = lf._weighted_design(Phi, B, sw)
    M = e.size
    N = A.shape[0]
    if ridge and ridge > 0.0:
        Apad = np.vstack([A, ridge * np.eye(M)])
    else:
        Apad = A
    try:
        Q, R1 = np.linalg.qr(Apad)
        if not np.all(np.isfinite(R1)):
            raise np.linalg.LinAlgError
        d = np.abs(np.diag(R1))
        if d.min() <= 1e-12 * max(d.max(), 1e-300):
            raise np.linalg.LinAlgError
        Qt = Q[:N]
        X = np.linalg.solve(R1, Qt.T @ Y)
        Res = A @ X
        Res -= Y
        if ridge and ridge > 0.0:
            Res = np.vstack([Res, ridge * X])
        return dict(theta=theta, A=A, Y=Y, Q=Q, R1=R1, X=X, Res=Res,
                    N=N, M=M, ridge=ridge, svd=False)
    except np.linalg.LinAlgError:
        X, _, _, _ = np.linalg.lstsq(Apad, np.vstack(
            [Y, np.zeros((M, Y.shape[1]))]), rcond=rcond)
        Res = Apad @ X
        Res -= np.vstack([Y, np.zeros((M, Y.shape[1]))])
        return dict(theta=theta, A=Apad, Y=None, Q=None, R1=None, X=X,
                    Res=Res, N=N, M=M, ridge=ridge, svd=True)


def factors_jac_a6(core, E, sw, rcond):
    M, N, ridge = core["M"], core["N"], core["ridge"]
    e, w = core["theta"][0::2], core["theta"][1::2]
    dPhi_de, dPhi_dw = lf.basis_derivatives(E, e, w)
    D = np.stack([dPhi_de, dPhi_dw], axis=2).reshape(
        dPhi_de.shape[0], 2 * M)
    if sw is not None:
        s = np.sqrt(np.asarray(sw, dtype=float))
        np.multiply(D, s[:, None], out=D)
    if not core["svd"]:
        Q = core["Q"]
        Qt, Qb = Q[:N], Q[N:]
        QtD = Qt.T @ D
        Zt = D - Qt @ QtD
        Zb = -(Qb @ QtD)
        Z = np.vstack([Zt, Zb]) if ridge and ridge > 0.0 else Zt
        K = solve_triangular(core["R1"], Q.T, check_finite=False).T
    else:
        Apad = core["A"]
        Dpad = np.vstack([D, np.zeros((M, 2 * M))])
        coef, _, _, _ = np.linalg.lstsq(Apad, Dpad, rcond=rcond)
        Z = Apad @ coef
        np.subtract(Dpad, Z, out=Z)
        pinvI, _, _, _ = np.linalg.lstsq(Apad, np.eye(Apad.shape[0]),
                                         rcond=rcond)
        K = pinvI.T
    return core["Res"], core["X"], Z, K, Z.T @ core["Res"]


rng = np.random.default_rng(17)
print("=== exact gate: a6 vs current ===")
for (N, M, n_ch, ridge) in [(189, 50, 12, 0.05), (189, 150, 12, 0.05),
                            (1001, 70, 19, 0.05), (451, 100, 12, 0.05),
                            (189, 50, 12, 0.0)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    ref = lf._varproj_factors(E, B, theta, sw, None, ridge)
    out = factors_jac_a6(factors_core_a6(E, B, theta, sw, None, ridge),
                        E, sw, None)
    exact = all(np.array_equal(a, b) for a, b in zip(ref, out))
    close = all(np.allclose(a, b, rtol=1e-12, atol=1e-14)
                for a, b in zip(ref, out))
    worst = max(np.max(np.abs(a - b)) for a, b in zip(ref, out))
    print(f"  N={N} M={M:3d} n_ch={n_ch} ridge={ridge}: "
          f"exact={exact} close={close} worst|diff|={worst:.2e}")

print("\n=== timing (full call, ridge path) ===")
for (N, M, n_ch) in [(189, 150, 12), (189, 50, 12), (451, 100, 12),
                     (1001, 70, 19)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    ts = {}
    for tag, fn in [
            ("cur", lambda: lf._varproj_factors(E, B, theta, sw, None, 0.05)),
            ("a6", lambda: factors_jac_a6(
                factors_core_a6(E, B, theta, sw, None, 0.05), E, sw, None))]:
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(50):
                fn()
            best = min(best, (time.perf_counter() - t0) / 50)
        ts[tag] = best * 1000
    print(f"  ({N:4d},{n_ch:3d},M={M:3d}): cur {ts['cur']:8.3f} ms  "
          f"a6 {ts['a6']:8.3f} ms  {ts['cur'] / ts['a6']:.2f}x")
