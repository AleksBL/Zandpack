"""Standalone test: does the matrix-free VarProjJacobian + lsmr path beat
the dense-jac + exact-SVD path on the RSSE-shaped problem?

Replicates fit_poles' two branches in isolation (NO AI_fit edits):
  dense:    jac returns the (N*r, 2M) array, x_scale='jac'
  operator: jac returns VarProjJacobian, x_scale=column norms
Same fun, same init, same bounds, method='trf'. Compares cost, nfev,
and wall time.
"""
import lzma
import os
import pickle
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined'))
sys.path.insert(0, ROOT)
import AI_fit as lf
sys.path.remove(ROOT)


def load(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)


R = load('C1.xz')
sys.path.insert(0, ROOT)
from refine_sampling import extract_dense
sys.path.remove(ROOT)
E_raw, F_raw = extract_dense(R, 0, 0)
fd = lf.prepare(E_raw, F_raw)
chans = lf.hermitian_channels(fd.F.shape[1])
B_full = lf.pack_responses(fd.F, chans)
Bc = lf.compress_channels(B_full)          # what production uses
E, B = fd.E, Bc
sw = np.ones(E.size)

rng = np.random.default_rng(7)
results = {}
for M in [24, 50, 100]:
    e0 = rng.uniform(-1, 1, M)
    w0 = rng.uniform(0.02, 0.08, M)
    bfun = lf.default_bounds(E, w_min=0.02, w_max=0.4)
    lo, hi = bfun(M)
    e0 = np.clip(e0, lo[0::2], hi[0::2])
    w0 = np.clip(w0, lo[1::2], hi[1::2])
    theta0 = np.empty(2 * M)
    theta0[0::2], theta0[1::2] = e0, w0

    def factors(theta):
        return lf._varproj_factors(E, B, theta, sw, None, 0.0)

    def fun(theta):
        Res, *_ = factors(theta)
        return Res.ravel(order="C")

    # dense branch (as in fit_poles)
    def jac_dense(theta):
        _, J = lf.residual_and_jacobian(E, B, theta, sw, None, 0.0)
        return J

    # operator branch (as in fit_poles)
    def jac_op(theta):
        Res, X, Z, K, S = factors(theta)
        return lf.VarProjJacobian(X, Z, K, S)

    Res, X, Z, K, S = factors(theta0)
    zz = np.sum(Z * Z, axis=0)
    ss = np.sum(S * S, axis=1)
    x2 = np.sum(X * X, axis=1)
    k2 = np.sum(K * K, axis=0)
    ZK = Z.T @ K
    XS = X @ S.T
    col2 = zz * np.repeat(x2, 2) + np.repeat(k2, 2) * ss
    col2[0::2] -= 2.0 * np.diag(ZK[0::2]) * np.diag(XS[:, 0::2])
    col2[1::2] -= 2.0 * np.diag(ZK[1::2]) * np.diag(XS[:, 1::2])
    x_scale = np.sqrt(np.maximum(col2, 1e-300))
    x_scale = np.maximum(x_scale, 1e-12 * np.max(x_scale))

    out = {}
    for tag, jac, xs in [("dense", jac_dense, "jac"),
                         ("operator", jac_op, x_scale)]:
        t0 = time.perf_counter()
        res = least_squares(fun, theta0, jac=jac, method="trf",
                            bounds=(lo, hi), x_scale=xs,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        out[tag] = (res.cost, res.nfev, time.perf_counter() - t0,
                    res.optimality)
        print(f"M={M:3d} {tag:8s}: cost {res.cost:.6e}  nfev {res.nfev:4d}  "
              f"wall {out[tag][2]:6.2f}s  opt {res.optimality:.2e}",
              flush=True)
    cd, co = out['dense'][0], out['operator'][0]
    print(f"  -> cost ratio operator/dense: {co / cd:.4f}  "
          f"speedup dense/op: {out['dense'][2] / out['operator'][2]:.2f}x"
          f"   quality {'OK' if co <= 1.05 * max(cd, 1e-30) else 'WORSE'}")
