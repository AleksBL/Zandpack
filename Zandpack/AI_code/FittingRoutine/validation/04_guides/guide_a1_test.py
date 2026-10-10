"""Guide A1+A3: lazy Jacobian split in _varproj_factors.

Variants:
  cur   : AI_fit._varproj_factors (current, everything every call)
  a1    : core (QR, X, Res) + lazy jac (D, Z, K, S); K via
          solve_triangular(..., check_finite=False)   [A3]

Checks:
  1. bit-identity of (Res, X, Z, K, S): a1(core+jac) vs cur — QR path,
     ridge path, and the rank-deficient SVD fallback path.
  2. realistic-pattern benchmark with the MEASURED 52.3% cost-only
     ratio: per accepted step (core+jac) vs per trial step (core only).
"""
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402


def factors_core(E, B, theta, sw, rcond, ridge):
    e, w = theta[0::2], theta[1::2]
    Phi = lf.lorentzian_basis(E, e, w)
    A, Y = lf._weighted_design(Phi, B, sw)
    M = e.size
    if ridge and ridge > 0.0:
        A = np.vstack([A, ridge * np.eye(M)])
        Y = np.vstack([Y, np.zeros((M, Y.shape[1]))])
    try:
        Q, R1 = np.linalg.qr(A)
        if not np.all(np.isfinite(R1)):
            raise np.linalg.LinAlgError
        d = np.abs(np.diag(R1))
        if d.min() <= 1e-12 * max(d.max(), 1e-300):
            raise np.linalg.LinAlgError
        X = np.linalg.solve(R1, Q.T @ Y)
        Res = A @ X
        Res -= Y
        return dict(theta=theta, A=A, Y=Y, Q=Q, R1=R1, X=X, Res=Res,
                    ridge=ridge, svd=False)
    except np.linalg.LinAlgError:
        X, _, _, _ = np.linalg.lstsq(A, Y, rcond=rcond)
        Res = A @ X
        Res -= Y
        return dict(theta=theta, A=A, Y=Y, Q=None, R1=None, X=X, Res=Res,
                    ridge=ridge, svd=True)


def build_D(E, theta, sw, ridge):
    e, w = theta[0::2], theta[1::2]
    M = e.size
    dPhi_de, dPhi_dw = lf.basis_derivatives(E, e, w)
    D = np.stack([dPhi_de, dPhi_dw], axis=2).reshape(
        dPhi_de.shape[0], 2 * M)
    if sw is not None:
        s = np.sqrt(np.asarray(sw, dtype=float))
        np.multiply(D, s[:, None], out=D)
    if ridge and ridge > 0.0:
        D = np.vstack([D, np.zeros((M, 2 * M))])
    return D


def factors_jac(core, E, sw, rcond):
    D = build_D(E, core["theta"], sw, core["ridge"])
    if not core["svd"]:
        Q, R1 = core["Q"], core["R1"]
        Z = Q @ (Q.T @ D)
        np.subtract(D, Z, out=Z)
        K = solve_triangular(R1, Q.T, check_finite=False).T
    else:
        A = core["A"]
        coef, _, _, _ = np.linalg.lstsq(A, D, rcond=rcond)
        Z = A @ coef
        np.subtract(D, Z, out=Z)
        pinvI, _, _, _ = np.linalg.lstsq(A, np.eye(A.shape[0]),
                                         rcond=rcond)
        K = pinvI.T
    return core["Res"], core["X"], Z, K, Z.T @ core["Res"]


rng = np.random.default_rng(17)
print("=== bit-identity: a1(core+jac) vs current ===")
ok = True
for (N, M, n_ch, ridge, dup) in [
        (189, 50, 12, 0.05, False), (189, 150, 12, 0.05, False),
        (189, 50, 12, 0.0, False), (1001, 70, 19, 0.05, False),
        (451, 100, 12, 0.05, False), (189, 50, 12, 0.05, True),
        (451, 60, 36, 0.05, True)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    if dup:
        theta[2] = theta[0]
        theta[3] = theta[1]
    sw = np.ones(N)
    ref = lf._varproj_factors(E, B, theta, sw, None, ridge)
    core = factors_core(E, B, theta, sw, None, ridge)
    out = factors_jac(core, E, sw, None)
    same = all(np.array_equal(a, b) for a, b in zip(ref, out))
    ok &= same
    print(f"  N={N} M={M:3d} n_ch={n_ch} ridge={ridge} "
          f"{'fallback' if dup else 'QR      '}: "
          f"{'bitwise identical' if same else 'DIFF'}")
print("IDENTITY:", "PASS" if ok else "FAIL")

print("\n=== realistic-pattern benchmark "
      "(52.3% cost-only / 47.7% fun+jac pairs) ===")
print(f"{'config':>22s} {'cur ms/iter':>11s} {'a1 ms/iter':>10s} "
      f"{'speedup':>8s}")
for (N, M, n_ch, ridge) in [(189, 50, 12, 0.05), (189, 150, 12, 0.05),
                            (1001, 70, 19, 0.05), (451, 100, 12, 0.05)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    sw = np.ones(N)
    thetas = []
    for k in range(60):
        theta = np.empty(2 * M)
        theta[0::2] = rng.uniform(-1, 1, M)
        theta[1::2] = rng.uniform(0.005, 0.1, M)
        thetas.append(theta)

    def cur_iter():
        for th in thetas:
            lf._varproj_factors(E, B, th, sw, None, ridge)

    def a1_iter():
        # 52.3% of calls cost-only (trial), 47.7% fun+jac pairs
        for i, th in enumerate(thetas):
            core = factors_core(E, B, th, sw, None, ridge)
            if i % 100 < 48:            # pair
                factors_jac(core, E, sw, None)

    for fn in (cur_iter, a1_iter):
        fn()                            # warm
    t = {}
    for tag, fn in [("cur", cur_iter), ("a1", a1_iter)]:
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t0) / len(thetas))
        t[tag] = best * 1000
    print(f"  ({N:4d},{n_ch:3d},M={M:3d}){'':4s} {t['cur']:10.3f} "
          f"{t['a1']:9.3f} {t['cur'] / t['a1']:7.2f}x")
