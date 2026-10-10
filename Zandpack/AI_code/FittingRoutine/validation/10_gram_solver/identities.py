"""never_form_J guide, validation step 1-2: Gram/gradient identities on
REAL factors from AI_fit's own varproj machinery.

For each problem (synthetic RSSE/ag3au3/nogap-shaped + REAL VerySmallRSSE
lead0/lead1 data, early and late iterates):
  - factors via AI_fit._varproj_core/_varproj_jac (QR path and FORCED
    SVD-fallback path)
  - dense J via AI_fit._assemble_dense_j
  - G, g via the guide's gram_and_grad (verbatim from the guide)
  - check |G - J^T J| / |J^T J| and |g - J^T f| / max(|g|, tiny)
Also records cond(J), the eigenvalue spectrum of G vs singular values of J,
and how many directions fall below CUT * s[0] (CUT=1e-7) - conditioning info
for the eigh truncation.
OMP_NUM_THREADS=1.
"""
import lzma
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import solve_triangular

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf  # noqa: E402
sys.path.remove(ROOT)


# ---------------- guide 3.1 verbatim (QR path branch kept) ----------------
def gram_and_grad(Res, X, Z, S, R1=None, Q=None, K=None):
    """G = J^T J (2M x 2M) and g = J^T f without forming J."""
    M = X.shape[0]
    k = np.arange(2 * M) // 2

    ZZ = Z.T @ Z
    XX = X @ X.T
    SS = S @ S.T
    if R1 is not None:
        Ri = solve_triangular(R1, np.eye(M), check_finite=False)
        KK = Ri @ Ri.T                          # K^T K = R1^-1 R1^-T
        KtRes = solve_triangular(R1, Q.T @ Res, check_finite=False)
    else:
        KK = K.T @ K
        KtRes = K.T @ Res

    G = ZZ * XX[np.ix_(k, k)] + KK[np.ix_(k, k)] * SS
    G = 0.5 * (G + G.T)                          # exact symmetry for eigh

    T = Res @ X.T                                # (R, M)
    g = np.einsum("np,np->p", Z, T[:, k])
    g -= np.einsum("pc,pc->p", S, KtRes[k])
    return G, g


def make_synth(N, M, n_ch, rng, ridge=0.05, sharp=True):
    """Random Lorentzian mixture sampled on E in [-1,1]: RSSE-shaped etc."""
    rng = np.random.default_rng(rng)
    E = np.linspace(-1.0, 1.0, N)
    e = rng.uniform(-0.9, 0.9, M)
    w = rng.uniform(0.01, 0.15, M) if sharp else rng.uniform(0.02, 0.4, M)
    Phi = lf.lorentzian_basis(E, e, w)
    Xtrue = rng.normal(size=(M, n_ch))
    B = Xtrue.T @ Phi.T                          # (n_ch, N)
    B += 0.01 * rng.normal(size=B.shape)
    theta = np.empty(2 * M)
    theta[0::2] = e
    theta[1::2] = w
    sw = 1.0 / np.linspace(0.5, 1.5, N)
    return E, B, theta, sw


def check_case(name, E, B, theta, sw, ridge, force_svd=False):
    th = theta.copy()
    if force_svd:
        # make A rank-deficient: exact duplicate pole -> QR rank check fires
        th[0], th[1] = th[2], th[3]
        name += " [SVD-fallback]"
    core = lf._varproj_core(E, B, th, sw, None, ridge=ridge)
    Z, K, S = lf._varproj_jac(core, E, sw, None)
    Res, X = core["Res"], core["X"]
    J = lf._assemble_dense_j(Res, X, Z, K, S)
    f = Res.ravel(order="C")
    if core["svd_path"]:
        G, g = gram_and_grad(Res, X, Z, S, K=K)
    else:
        G, g = gram_and_grad(Res, X, Z, S, R1=core["R1"], Q=core["Q"])

    JtJ = J.T @ J
    Jtf = J.T @ f
    nG = np.linalg.norm(JtJ)
    errG = np.linalg.norm(G - JtJ) / nG
    errg = np.linalg.norm(g - Jtf) / max(np.linalg.norm(Jtf), 1e-300)

    # conditioning info: sv of J vs eigenvalues of G; CUT truncation count
    sv = np.linalg.svd(J, compute_uv=False)
    ev = np.linalg.eigvalsh(G)
    ev = ev[::-1]
    # eig(G) should equal sv(J)^2 (G = J^T J)
    err_ev = np.max(np.abs(np.sqrt(np.maximum(ev, 0)) - sv)) / sv[0]
    CUT = 1e-7
    n_trunc = int(np.sum(sv < CUT * sv[0]))
    cond = sv[0] / sv[-1]
    print(f"{name:38s} G err {errG:.2e}  g err {errg:.2e}  "
          f"eig/sv err {err_ev:.2e}  cond(J) {cond:.1e}  "
          f"trunc<{CUT:.0e}: {n_trunc}/{sv.size}")
    assert errG < 1e-11, name
    assert errg < 1e-11, name
    return errG, errg


def main():
    # synthetic, guide-section-1 shapes (R = N+M rows incl ridge, n = 2M)
    cases = [
        ("synth RSSE-like R=264 n=150", 189, 75, 6, 11),
        ("synth ag3au3-like R=551 n=200", 451, 100, 12, 12),
        ("synth nogap-like R=1071 n=140", 1001, 70, 12, 13),
    ]
    for name, N, M, n_ch, rng in cases:
        E, B, theta, sw = make_synth(N, M, n_ch, rng)
        check_case(name, E, B, theta, sw, ridge=0.05)
        check_case(name, E, B, theta, sw, ridge=0.05, force_svd=True)

    # real VerySmallRSSE data, early (linspace init) and late (short fit)
    refined = os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined')
    os.chdir(refined)
    with lzma.open('C1.xz', 'rb') as fh:
        R = pickle.load(fh)
    sys.path.insert(0, ROOT)
    from refine_sampling import extract_dense  # noqa: E402
    sys.path.remove(ROOT)

    for lead in (0, 1):
        E_raw, F_raw = extract_dense(R, lead, 0)
        fd = lf.prepare(E_raw, F_raw)
        chans = lf.hermitian_channels(fd.F.shape[1])
        B = lf.pack_responses(fd.F, chans)
        p = fd.F.shape[1]
        n_ch = B.shape[0]
        sw = lf._sample_weights(fd.F, "relative", quad=lf._quad_widths(fd.E))
        N = fd.E.size
        for tag, n_init in (("early n=24", 24), ("late-ish n=59", 59)):
            e0, w0 = lf.linspace_init(fd.E, n_init)
            res = lf.fit_poles(fd.E, B, e0, w0, sample_weights=sw,
                               ridge=0.0, bounds_fun=lf.default_bounds(fd.E),
                               max_nfev=200, tr_solver="lsmr")
            th = np.empty(2 * res["e"].size)
            th[0::2] = res["e"]
            th[1::2] = res["w"]
            check_case(f"RSSE lead{lead} p={p} n_ch={n_ch} {tag}",
                       fd.E, B, th, sw, ridge=0.05)
            check_case(f"RSSE lead{lead} p={p} n_ch={n_ch} {tag}",
                       fd.E, B, th, sw, ridge=0.05, force_svd=True)
    print("ALL IDENTITY CHECKS PASSED")


if __name__ == "__main__":
    main()
