"""Benchmark variants for the dense Jacobian assembly in
residual_and_jacobian (lines 445-449):
  a. current numpy (repeat + 3-index broadcast + transpose copy)
  b. preallocated repeat buffers (np.empty + strided fill) + broadcast
  c. numba kernel writing J directly (no Xr/Kr, no broadcast temps)

Checks BITWISE identity of (r, J) for every variant vs (a), then times
each at production shapes. Ridge cases included (Z/K have N+M rows).
"""
import sys
import time

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
from numba import njit  # noqa: E402


@njit(cache=True)
def _jac_kernel(Z, X, K, S, J):
    n_rows, twoM = Z.shape
    n_ch = X.shape[1]
    for n in range(n_rows):
        base = n * n_ch
        for c in range(n_ch):
            row = base + c
            for j in range(twoM):
                i = j // 2
                J[row, j] = Z[n, j] * X[i, c] - K[n, i] * S[j, c]


def variant_a(E, B, theta, sw, rcond, ridge):
    return lf.residual_and_jacobian(E, B, theta, sw, rcond, ridge)


def variant_b(E, B, theta, sw, rcond, ridge):
    Res, X, Z, K, S = lf._varproj_factors(E, B, theta, sw, rcond,
                                          ridge=ridge)
    n_ch = Res.shape[1]
    M = X.shape[0]
    Xr = np.empty((2 * M, n_ch))
    Xr[0::2] = X
    Xr[1::2] = X
    Kr = np.empty((Z.shape[0], 2 * M))
    Kr[:, 0::2] = K
    Kr[:, 1::2] = K
    J = Z[:, :, None] * Xr[None, :, :] - Kr[:, :, None] * S[None, :, :]
    J = np.ascontiguousarray(J.transpose(0, 2, 1))
    return Res.ravel(order="C"), J.reshape(Res.shape[0] * n_ch, 2 * M)


def variant_c(E, B, theta, sw, rcond, ridge):
    Res, X, Z, K, S = lf._varproj_factors(E, B, theta, sw, rcond,
                                          ridge=ridge)
    n_ch = Res.shape[1]
    M = X.shape[0]
    J = np.empty((Res.shape[0] * n_ch, 2 * M))
    _jac_kernel(Z, X, K, S, J)
    return Res.ravel(order="C"), J


rng = np.random.default_rng(5)

print("=== bitwise identity vs current ===")
ok = True
for (N, M, n_ch, ridge) in [(189, 24, 12, 0.0), (189, 50, 12, 0.05),
                            (189, 100, 12, 0.0), (451, 60, 36, 0.05),
                            (100, 8, 5, 0.0)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    ra, Ja = variant_a(E, B, theta, sw, None, ridge)
    rb, Jb = variant_b(E, B, theta, sw, None, ridge)
    rc, Jc = variant_c(E, B, theta, sw, None, ridge)
    sb = np.array_equal(Ja, Jb) and np.array_equal(ra, rb)
    sc = np.array_equal(Ja, Jc) and np.array_equal(ra, rc)
    ok &= sb and sc
    print(f"  N={N} M={M} n_ch={n_ch} ridge={ridge}: "
          f"prealloc={'identical' if sb else 'DIFF'}  "
          f"numba={'identical' if sc else 'DIFF'}")
print("IDENTITY:", "PASS" if ok else "FAIL")

# warm up numba (compile) before timing
E = np.sort(rng.uniform(-1, 1, 189))
B = rng.normal(size=(12, 189))
theta = np.empty(100)
theta[0::2] = rng.uniform(-1, 1, 50)
theta[1::2] = rng.uniform(0.005, 0.1, 50)
variant_c(E, B, theta, None, None, 0.05)

print("\n=== timing (min of reps, ms/call) ===")
print(f"{'shape (N, n_ch, M)':>24s} {'a numpy':>9s} {'b preall':>9s} "
      f"{'c numba':>9s} {'c/a':>6s}")
for (N, n_ch, M) in [(189, 12, 24), (189, 12, 50), (189, 12, 100),
                     (451, 36, 60), (451, 378, 60), (1001, 27, 60)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    reps = {v: np.inf for v in "abc"}
    for tag, fn in [("a", variant_a), ("b", variant_b),
                    ("c", variant_c)]:
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(100):
                fn(E, B, theta, sw, None, 0.05)
            best = min(best, (time.perf_counter() - t0) / 100)
        reps[tag] = best * 1000
    print(f"  ({N:4d},{n_ch:3d},{M:3d}){'':6s} {reps['a']:8.3f} "
          f"{reps['b']:8.3f} {reps['c']:8.3f} {reps['a'] / reps['c']:5.2f}x")
