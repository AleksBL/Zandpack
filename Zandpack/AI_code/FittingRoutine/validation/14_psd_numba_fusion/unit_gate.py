"""Unit gate for the PSD/ADMM numba fusion (project_psd fused
Hermitize+reconstruct kernels; the ADMM iteration's fused X-update
RHS, over-relaxation, and in-place U update with reused workspaces).

Gates:
1. kernel references - hermitize / ghat / c_hat must be BITWISE
   identical to the numpy sequences; reconstruct is a k-reduction
   (bitwise against einsum recorded, allclose required).
2. project_psd: new vs the pre-change module (backup
   AI_fit_2026-10-09_pre_psd_numba.py) on random Hermitian /
   non-Hermitian / REAL production residues.
3. psd_solve_admm: new vs old module on REAL production inputs
   (Ag3Au3 production fits, full packing incl. tail anchors):
   final residues, packed values, objective, iteration count,
   wall.

Run twice: with numba (default) and NUMBA_DISABLE_JIT=1 (fallback
paths). OMP_NUM_THREADS=1.
"""
import gzip
import importlib.util
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, os.path.join(ROOT, "validation",
                                "11_gram_basin_spread"))
import spread_common as sc  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    os.chdir("/tmp")
    new = sc.import_ai_fit()
    old = load_module(os.path.join(ROOT, "backups",
                                   "AI_fit_2026-10-09_pre_psd_numba.py"),
                      "ai_fit_old")
    rng = np.random.default_rng(20261009)
    n_fail = 0
    print(f"[gate] numba active: {new._njit is not None}", flush=True)

    # ---- 1. kernel references ---------------------------------------
    for M, p in [(1, 1), (2, 3), (17, 6), (100, 27)]:
        C = rng.standard_normal((M, p, p)) + \
            1j * rng.standard_normal((M, p, p))
        out = np.empty_like(C)
        if new._njit is not None:
            new._psd_hermitize_kernel(C, out)
            ref = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))
            ok = np.array_equal(out, ref)
            print(f"  hermitize M={M} p={p}: "
                  f"{'BITWISE' if ok else 'MISMATCH'}", flush=True)
            n_fail += not ok

            lam = rng.standard_normal((M, p))
            V = rng.standard_normal((M, p, p)) + \
                1j * rng.standard_normal((M, p, p))
            out2 = np.empty_like(V)
            new._psd_reconstruct_kernel(V, lam, out2)
            ref2 = np.einsum("mik,mk,mjk->mij", V, lam, np.conj(V))
            bit = np.array_equal(out2, ref2)
            rel = float(np.max(np.abs(out2 - ref2)) /
                        max(float(np.abs(ref2).max()), 1e-300))
            ok = bit or rel < 1e-13
            print(f"  reconstruct M={M} p={p}: "
                  f"{'BITWISE' if bit else f'allclose rel {rel:.2e}'} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            n_fail += not ok

            chans = new.hermitian_channels(p)
            ia, ib, re = new.channel_indices(chans)
            Z = rng.standard_normal((M, p, p)) + \
                1j * rng.standard_normal((M, p, p))
            U = rng.standard_normal((M, p, p)) + \
                1j * rng.standard_normal((M, p, p))
            AtY = rng.standard_normal((M, len(chans)))
            rho = 0.7
            g = np.empty_like(AtY)
            new._admm_ghat_kernel(Z, U, ia, ib, re, AtY, rho, g)
            dvals = (Z - U)[:, ia, ib]
            Gs = np.where(re[None, :], dvals.real, dvals.imag)
            ref3 = AtY + rho * Gs
            ok = np.array_equal(g, ref3)
            print(f"  admm ghat M={M} p={p}: "
                  f"{'BITWISE' if ok else 'MISMATCH'}", flush=True)
            n_fail += not ok

            alpha = 1.6
            ch = np.empty_like(Z)
            new._admm_c_hat_kernel(C, Z, alpha, ch)
            ref4 = alpha * C + (1.0 - alpha) * Z
            ok = np.array_equal(ch, ref4)
            print(f"  admm c_hat M={M} p={p}: "
                  f"{'BITWISE' if ok else 'MISMATCH'}", flush=True)
            n_fail += not ok

    # ---- 2. project_psd: new vs old module --------------------------
    print("--- project_psd new vs old ---", flush=True)
    for tag, C in [("rand hermitian", None), ("rand dense", None),
                   ("real l1 residues", "real")]:
        if C == "real":
            with gzip.open(os.path.join(ROOT, "Data2", "Ag3Au3",
                                       "tmp.gz"), "rb") as f:
                R = pickle.load(f)
            E, F = new.extract_dense(R, 1, 0)
            d = np.load(os.path.join(
                ROOT, "validation", "11_gram_basin_spread",
                "ag3au3_gram_production_fits.npz"),
                allow_pickle=True)
            fit = d["lead1_pick"].item()
            fd = new.prepare(E, F)
            Cmat = np.asarray(fit.C) / fd.f_scale
            Cmat = Cmat[:20]
        else:
            L = rng.standard_normal((20, 27, 27)) + \
                1j * rng.standard_normal((20, 27, 27))
            Cmat = np.einsum("mik,mjk->mij", L, np.conj(L)) \
                if tag == "rand hermitian" else \
                L + 1j * np.swapaxes(L, 1, 2)
        t0 = time.perf_counter()
        cn, ccn, wn = new.project_psd(Cmat)
        tn = time.perf_counter() - t0
        t0 = time.perf_counter()
        co, cco, wo = old.project_psd(Cmat)
        to = time.perf_counter() - t0
        bit = np.array_equal(cn, co)
        rel = float(np.max(np.abs(cn - co)) /
                    max(float(np.abs(co).max()), 1e-300))
        ok = (ccn == cco and wn == wo) and (bit or rel < 1e-12)
        print(f"  {tag:18s}: {'BITWISE' if bit else f'rel {rel:.2e}'} "
              f"clipped {ccn}/{cco} worst {wn:+.3g}/{wo:+.3g} "
              f"wall {to*1e3:.1f}ms -> {tn*1e3:.1f}ms "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        n_fail += not ok

    # ---- 3. psd_solve_admm: new vs old on REAL inputs ---------------
    print("--- psd_solve_admm new vs old (real inputs) ---", flush=True)
    with gzip.open(os.path.join(ROOT, "Data2", "Ag3Au3", "tmp.gz"),
                   "rb") as f:
        R = pickle.load(f)
    sys.path.insert(0, ROOT)
    import refine_sampling  # noqa: F401
    sys.path.remove(ROOT)
    d = np.load(os.path.join(ROOT, "validation", "11_gram_basin_spread",
                             "ag3au3_gram_production_fits.npz"),
                allow_pickle=True)
    for lead in (0, 1):
        fit = d[f"lead{lead}_pick"].item()
        E, F = new.extract_dense(R, lead, 0)
        fd = new.prepare(E, F)
        p = F.shape[1]
        chans = new.hermitian_channels(p)
        B = new.pack_responses(fd.F, chans)
        span = float(fd.E.max() - fd.E.min())
        dd = np.array([0.05, 0.15, 0.35, 0.65]) * 0.5 * span
        E_fit = np.concatenate([fd.E, fd.E.min() - dd,
                                 fd.E.max() + dd])
        B_fit = np.concatenate([B, np.zeros((B.shape[0], 8))],
                               axis=1)
        Phi = new.lorentzian_basis(
            E_fit, fd.scale_centers(np.asarray(fit.e)),
            np.asarray(fit.w) / fd.e_scale)
        for iters in (50, 200):
            t0 = time.perf_counter()
            Cn, Xn = new.psd_solve_admm(Phi, B_fit, chans, p,
                                         iters=iters)
            tn = time.perf_counter() - t0
            t0 = time.perf_counter()
            Co, Xo = old.psd_solve_admm(Phi, B_fit, chans, p,
                                         iters=iters)
            to = time.perf_counter() - t0
            rel = float(np.max(np.abs(Cn - Co)) /
                        max(float(np.abs(Co).max()), 1e-300))
            relx = float(np.max(np.abs(Xn - Xo)) /
                         max(float(np.abs(Xo).max()), 1e-300))
            ok = rel < 1e-10 and relx < 1e-10
            print(f"  lead{lead} iters={iters}: C rel {rel:.2e} "
                  f"X rel {relx:.2e} wall {to:.2f}s -> {tn:.2f}s "
                  f"({to/max(tn,1e-9):.1f}x) "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            n_fail += not ok

    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}",
          flush=True)


if __name__ == "__main__":
    main()
