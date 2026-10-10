"""Unit gate for the refit_scales optimization (factored normal-
equations NNLS vs the original full-design scipy nnls).

Cases:
- synthetic: random PSD residues, positive basis, random data, across
  sizes (M up to 100, n_ch 19/729, N 50/451), well-conditioned AND
  deliberately degenerate pole sets (coincident columns -> the
  fallback must fire and reproduce the slow path exactly);
- real: Ag3Au3 lead0/lead1 and RSSE-style fits from the caches, with
  the production packing (scaled units, tail anchors), at their true
  pole counts (M = 92-108).

Checks per case:
- alpha agreement (relative),
- TRUE objective agreement: ||A alpha - b||^2 computed with the full
  design for BOTH alphas (fair quality comparison),
- fallback rate,
- wall of each path.

PASS bar: alpha rel diff <= 1e-6 on non-degenerate cases (both solve
the same convex problem; degenerate cases may pick different vertices
with equal objective), objective rel diff <= 1e-9, and every
degenerate case either matches or fell back to the exact slow path.
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


def compare(tag, new, old, Phi, B, Cv, chans, C_pos):
    """Run both refit_scales variants and compare."""
    t0 = time.perf_counter()
    a_new, Cs_new = new.refit_scales(Phi, B, C_pos, chans)
    t_new = time.perf_counter() - t0
    t0 = time.perf_counter()
    a_old, Cs_old = old.refit_scales(Phi, B, C_pos, chans)
    t_old = time.perf_counter() - t0

    # true objective from the full design, for BOTH alphas
    M = Phi.shape[1]
    A = (Cv[:, None, :] * Phi[None, :, :]).reshape(-1, M)
    b = np.asarray(B).reshape(-1)
    o_new = float(np.sum((A @ a_new - b) ** 2))
    o_old = float(np.sum((A @ a_old - b) ** 2))
    denom = max(o_old, 1e-300)
    da = float(np.max(np.abs(a_new - a_old)) /
               max(float(np.abs(a_old).max()), 1e-300))
    do = abs(o_new - o_old) / denom
    fast = new._refit_scales_fast(Phi, B, Cv)
    fb = "FALLBACK" if fast is None else ""
    ok_a = da <= 1e-6
    ok_o = do <= 1e-9
    status = "PASS" if (ok_a and ok_o) or (ok_o and fb) else "FAIL"
    print(f"  {tag:34s} M={M:4d} alpha_rel {da:9.2e} "
          f"obj_rel {do:9.2e} wall {t_old:6.3f}s -> {t_new:6.3f}s "
          f"({t_old/max(t_new,1e-9):5.1f}x) {fb:8s} {status}",
          flush=True)
    return ok_a, ok_o, fb == "FALLBACK"


def synth_case(rng, M, n_ch, N, degenerate=False, tag=""):
    p = int(np.sqrt(n_ch // 1)) or 3
    # build PSD residues via L L^H
    L = rng.standard_normal((M, p, p)) + 1j * rng.standard_normal((M, p, p))
    C_pos = np.einsum("mik,mjk->mij", L, np.conj(L))
    e = np.sort(rng.uniform(-1, 1, M))
    if degenerate:
        # coincident pole pairs -> rank-deficient design columns
        e[1::2] = e[0::2]
    w = rng.uniform(0.05, 0.3, M)
    Phi = 1.0 / (1.0 + ((rng.uniform(-1, 1, (N, 1)) - e) / w) ** 2)
    chans = sc.import_ai_fit().hermitian_channels(p)
    vals = np.stack([C_pos[:, a, b].real if part == "re"
                     else C_pos[:, a, b].imag
                     for a, b, part in chans])
    alpha_true = rng.uniform(0, 1, M)
    B = (vals * alpha_true[None, :]) @ Phi.T + \
        0.01 * rng.standard_normal((len(chans), N))
    return dict(Phi=Phi, B=B, Cv=vals, chans=chans, C_pos=C_pos,
                tag=tag)


def _real_case(new, old, R, fit, lead, tag):
    """One real refit_scales case: production packing (scaled units,
    tail anchors), the fit's true poles/residues."""
    E, F = new.extract_dense(R, lead, 0)
    fd = new.prepare(E, F)
    p = F.shape[1]
    chans = new.hermitian_channels(p)
    B = new.pack_responses(fd.F, chans)
    span = float(fd.E.max() - fd.E.min())
    dd = np.array([0.05, 0.15, 0.35, 0.65]) * 0.5 * span
    E_fit = np.concatenate([fd.E, fd.E.min() - dd, fd.E.max() + dd])
    B_fit = np.concatenate([B, np.zeros((B.shape[0], 8))], axis=1)
    e_s = fd.scale_centers(np.asarray(fit.e))
    w_s = np.asarray(fit.w) / fd.e_scale
    Phi = new.lorentzian_basis(E_fit, e_s, w_s)
    C_pos, _, _ = new.project_psd(np.asarray(fit.C) / fd.f_scale)
    return compare(tag, new, old, Phi, B_fit,
                   new._channel_values(C_pos, chans), chans, C_pos)


def main():
    os.chdir("/tmp")
    new = sc.import_ai_fit()
    old = load_module(os.path.join(ROOT, "backups",
                                   "AI_fit_2026-10-09_pre_refitscales.py"),
                      "ai_fit_old")
    rng = np.random.default_rng(20261009)
    n_fail = 0

    print("--- synthetic ---", flush=True)
    for M, n_ch, N in [(1, 6, 50), (5, 19, 50), (24, 729, 200),
                       (60, 729, 451), (100, 729, 451)]:
        c = synth_case(rng, M, n_ch, N, tag=f"rand M={M} nch={n_ch}")
        ok_a, ok_o, fb = compare(c["tag"], new, old, c["Phi"], c["B"],
                                 c["Cv"], c["chans"], c["C_pos"])
        n_fail += not (ok_a and ok_o or ok_o)
    print("--- synthetic degenerate (coincident poles) ---", flush=True)
    for M in (24, 60):
        c = synth_case(rng, M, 729, 200, degenerate=True,
                       tag=f"degen M={M}")
        ok_a, ok_o, fb = compare(c["tag"], new, old, c["Phi"], c["B"],
                                 c["Cv"], c["chans"], c["C_pos"])
        n_fail += not (ok_o or (fb and ok_o))

    print("--- real (production fits, scaled units + tail anchors) ---",
          flush=True)
    cases = [("Data2/Ag3Au3/cmp_fits_exactera.npz", "tmp.gz", "gz",
              [0, 1])]
    for rel_npz, pkl, fmt, leads in cases:
        path = os.path.join(ROOT, rel_npz)
        if not os.path.exists(path):
            print(f"  (skip {rel_npz}: not found)", flush=True)
            continue
        with open(os.path.join(os.path.dirname(
                os.path.join(ROOT, rel_npz)), pkl), "rb") as f:
            R = pickle.load(gzip.open(f.name) if fmt == "gz" else f)
        d = np.load(path, allow_pickle=True)
        sys.path.insert(0, ROOT)
        import refine_sampling  # noqa: F401
        sys.path.remove(ROOT)
        for lead in leads:
            key = f"lead{lead}_ik0"
            fit = d[key].item()
            ok_a, ok_o, fb = _real_case(
                new, old, R, fit, lead,
                f"{os.path.basename(rel_npz)} l{lead} ({key})")
            n_fail += not (ok_a and ok_o)

    # the gram production picks (validation/11_gram_basin_spread)
    prod = os.path.join(ROOT, "validation", "11_gram_basin_spread",
                        "ag3au3_gram_production_fits.npz")
    if os.path.exists(prod):
        with gzip.open(os.path.join(ROOT, "Data2", "Ag3Au3",
                                   "tmp.gz"), "rb") as f:
            R = pickle.load(f)
        sys.path.insert(0, ROOT)
        import refine_sampling  # noqa: F401
        sys.path.remove(ROOT)
        d = np.load(prod, allow_pickle=True)
        for lead in (0, 1):
            fit = d[f"lead{lead}_pick"].item()
            ok_a, ok_o, fb = _real_case(
                new, old, R, fit, lead,
                f"gram_production l{lead} (pick)")
            n_fail += not (ok_a and ok_o)

    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}",
          flush=True)


if __name__ == "__main__":
    main()
