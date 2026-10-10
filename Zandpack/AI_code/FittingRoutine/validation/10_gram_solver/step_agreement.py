"""never_form_J guide, validation steps 4-5: trust-region step agreement
between the Gram (eigh) path and scipy's exact SVD path, on REAL RSSE
factors, with active bounds (nonzero diag_h), several Delta, and the
forced SVD-fallback path. Also compares the full select_step (bounds
reflection machinery) with dense J_h vs G_h.
"""
import lzma
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import svd
from scipy.optimize._lsq.common import solve_lsq_trust_region

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
sys.path.remove(ROOT)


def gram_step(G, g, d, diag_h, Delta, cut=1e-7):
    """The _trf_gram inner step: eigh(G_aug) + truncated solve."""
    g_h = d * g
    G_h = G * np.outer(d, d)
    G_aug = G_h + np.diag(diag_h)
    G_aug = 0.5 * (G_aug + G_aug.T)
    lam, V = np.linalg.eigh(G_aug)
    lam = lam[::-1]
    V = V[:, ::-1]
    s = np.sqrt(np.maximum(lam, 0.0))
    keep = s > cut * s[0]
    if not np.any(keep):
        keep[0] = True
        s[0] = max(s[0], np.finfo(float).tiny)
    sk = s[keep]
    Vk = V[:, keep]
    uf = (Vk.T @ g_h) / sk
    p_h, _a, _i = solve_lsq_trust_region(int(keep.sum()), 10**9, uf, sk,
                                         Vk, Delta)
    return p_h, int((~keep).sum())


def exact_step(J, f, d, diag_h, Delta):
    """scipy trf_bounds' exact step: SVD of the augmented Jacobian."""
    g_h = (J * d).T @ f
    m, n = J.shape
    J_aug = np.empty((m + n, n))
    J_aug[:m] = J * d
    J_aug[m:] = np.diag(diag_h**0.5)
    f_aug = np.zeros(m + n)
    f_aug[:m] = f
    U, s, Vt = svd(J_aug, full_matrices=False)
    V = Vt.T
    uf = U.T @ f_aug
    p_h, _a, _i = solve_lsq_trust_region(n, m, uf, s, V, Delta)
    return p_h, 0


def check_agreement(name, E, B, theta, sw, ridge, force_svd=False):
    th = theta.copy()
    if force_svd:
        th[0], th[1] = th[2], th[3]
        name += " [SVD-fallback]"
    core = lf._varproj_core(E, B, th, sw, None, ridge=ridge)
    Z, K, S = lf._varproj_jac(core, E, sw, None)
    Res, X = core["Res"], core["X"]
    J = lf._assemble_dense_j(Res, X, Z, K, S)
    f = Res.ravel(order="C")
    if core["svd_path"]:
        G, g = lf.gram_and_grad(Res, X, Z, S, K=K)
    else:
        G, g = lf.gram_and_grad(Res, X, Z, S, R1=core["R1"], Q=core["Q"])

    # x near the w-bound so Coleman-Li diag_h is nonzero (active bounds)
    n = theta.size
    lo = np.empty(n)
    hi = np.empty(n)
    lo[0::2] = E.min() - 0.05 * (E.max() - E.min())
    hi[0::2] = E.max() + 0.05 * (E.max() - E.min())
    lo[1::2] = 0.02
    hi[1::2] = 0.5
    x = th.copy()
    x[1::2] = lo[1::2] + 1e-4 * np.abs(hi[1::2] - lo[1::2])  # on the bound

    v, dv = lf.CL_scaling_vector(x, g, lo, hi)
    scale = np.ones(n)
    scale_inv = 1.0 / scale
    v[dv != 0] *= scale_inv[dv != 0]
    d = v**0.5 * scale
    diag_h = g * dv * scale
    g_h = d * g

    # GN step norm for Delta scaling
    J_aug = np.vstack([J * d, np.diag(diag_h**0.5)])
    s_all = svd(J_aug, compute_uv=False)
    gn = np.linalg.norm((J * d).T @ f) / s_all[0]**2  # rough scale
    deltas = [0.05, 0.2, 1.0, 5.0]
    worst = 0.0
    truncs = 0
    for fac in deltas:
        Delta = fac * max(gn, 1e-12)
        p_ex, _ = exact_step(J, f, d, diag_h, Delta)
        p_gr, trunc = gram_step(G, g, d, diag_h, Delta)
        truncs += trunc
        if np.linalg.norm(p_ex) > 0:
            err = np.linalg.norm(p_gr - p_ex) / np.linalg.norm(p_ex)
        else:
            err = np.linalg.norm(p_gr - p_ex)
        worst = max(worst, err)

    # full select_step comparison (reflection/anti-gradient machinery)
    from scipy.optimize._lsq.trf import select_step as scipy_select_step
    J_h = J * d
    G_h = G * np.outer(d, d)
    sel_err = 0.0
    for fac in (0.3, 1.0, 3.0):
        Delta = fac * max(gn, 1e-12)
        p_ex, _ = exact_step(J, f, d, diag_h, Delta)
        p_gr, _ = gram_step(G, g, d, diag_h, Delta)
        for p_in, tag in ((p_ex, "from-exact"), (p_gr, "from-gram")):
            p = d * p_in
            a1 = scipy_select_step(x, J_h, diag_h, g_h, p.copy(),
                                   p_in.copy(), d, Delta, lo, hi, 0.995)
            a2 = lf._gram_select_step(x, G_h, diag_h, g_h, p.copy(),
                                      p_in.copy(), d, Delta, lo, hi, 0.995)
            e = np.max(np.abs(np.asarray(a1[0]) - np.asarray(a2[0])))
            e = max(e, abs(a1[2] - a2[2]))
            sel_err = max(sel_err, e / max(1.0, np.abs(a1[2])))

    print(f"{name:44s} step rel-diff max {worst:.2e} (Delta 0.05-5xGN)  "
          f"select_step rel-diff {sel_err:.2e}  truncated dirs {truncs}")
    return worst, sel_err


def main():
    refined = os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined')
    os.chdir(refined)
    with lzma.open('C1.xz', 'rb') as fh:
        R = pickle.load(fh)
    sys.path.insert(0, ROOT)
    from refine_sampling import extract_dense  # noqa: E402
    sys.path.remove(ROOT)

    worst_all = 0.0
    sel_worst = 0.0
    for lead in (0, 1):
        E_raw, F_raw = extract_dense(R, lead, 0)
        fd = lf.prepare(E_raw, F_raw)
        chans = lf.hermitian_channels(fd.F.shape[1])
        B = lf.pack_responses(fd.F, chans)
        sw = lf._sample_weights(fd.F, "relative", quad=lf._quad_widths(fd.E))
        for n_init in (24, 59):
            e0, w0 = lf.linspace_init(fd.E, n_init)
            res = lf.fit_poles(fd.E, B, e0, w0, sample_weights=sw,
                               ridge=0.0, bounds_fun=lf.default_bounds(fd.E),
                               max_nfev=200, tr_solver="lsmr")
            th = np.empty(2 * res["e"].size)
            th[0::2] = res["e"]
            th[1::2] = res["w"]
            for fs in (False, True):
                w, s = check_agreement(
                    f"RSSE lead{lead} n={n_init}", fd.E, B, th, sw,
                    0.05, force_svd=fs)
                worst_all = max(worst_all, w)
                sel_worst = max(sel_worst, s)
    print(f"\nWORST step rel-diff: {worst_all:.2e}   "
          f"WORST select_step rel-diff: {sel_worst:.2e}")
    assert worst_all < 5e-6, "step agreement FAILED"
    assert sel_worst < 1e-10, "select_step agreement FAILED"
    print("STEP AGREEMENT PASSED")


if __name__ == "__main__":
    main()
