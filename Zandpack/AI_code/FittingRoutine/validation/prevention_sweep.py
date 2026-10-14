#!/usr/bin/env python3
"""Prevention-measure calibration (2026-10-13 review round).

PART 1 - Tikhonov lambda sweep (closed form): one eigendecomposition
of the weighted+anchored design Gram at the monster fits' poles gives
X(lambda) for every lambda; find the smallest lambda that removes the
residue blow-up at the least rms cost (reviewer: the ridge is an
absolute 0.05 on entries of order 1 while sigma_max grows with N, so
it damps a large share of the spectrum; a RELATIVE filter is wanted).

PART 2 - sep_alpha sweep: fit() with the separation constraint at
alpha in {None, 0.05, 0.1, 0.2} on the two monster cases (Fig2Ag5
tgt1e4-class config, Ag3Au3 m1_w03 config), stab_weight on by
default; report sample rms, frob, cap ratio/flags, cancellation
index, and the per-band frobenius rms (the 2-4 eV band is the
reviewer's success metric). Plus one stab_weight=0 control on Ag3Au3.

Run:  OMP_NUM_THREADS=1 python validation/prevention_sweep.py
Out:  validation/prevention_sweep.json
"""
import gzip
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
OUT = os.path.join(ROOT, "validation", "prevention_sweep.json")
T0 = time.time()


def log(msg):
    print(f"[prevsweep {time.time()-T0:6.0f}s] {msg}", flush=True)


def load_all():
    _path0 = list(sys.path)
    R_f2 = pickle.load(gzip.open(
        ROOT + "/Data2/Fig2Ag5/tmp_refined_deriv.gz", "rb"))
    R_ag = pickle.load(gzip.open(
        ROOT + "/Data2/Ag3Au3/tmp_refined_deriv.gz", "rb"))
    for q in [q for q in sys.path if q not in _path0]:
        sys.path.remove(q)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base", ROOT + "/AI_fit_base.py")
    lf = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base"] = lf
    spec.loader.exec_module(lf)
    # old-era caches were pickled under the flat 'AI_fit' alias
    sys.modules.setdefault("AI_fit", lf)
    sys.modules.setdefault("lorentz_fit", lf)
    return lf, R_f2, R_ag


def weighted_design(lf, E, F, e, w):
    """(G, A, Y, scale): anchored+weighted design Gram and packed
    responses in scaled units, at the given (raw) poles."""
    fd = lf.prepare(E, F)
    E_s, F_s = fd.E, fd.F
    chans = lf.hermitian_channels(F_s.shape[1])
    B = lf.pack_responses(F_s, chans)
    qw = lf._quad_widths(E_s)
    sw = lf._sample_weights(F_s, "relative", quad=qw)
    span = float(E_s.max() - E_s.min())
    d = np.array([0.05, 0.15, 0.35, 0.65]) * 0.5 * span
    E_anchor = np.concatenate([E_s.min() - d, E_s.max() + d])
    E_aug = np.concatenate([E_s, E_anchor])
    Phi = lf.lorentzian_basis(E_aug, fd.scale_centers(e),
                              np.asarray(w) / fd.e_scale)
    if sw is None:
        swv = np.ones(E_aug.size)
    else:
        swv = np.concatenate([sw, np.full(E_anchor.size,
                                          0.05 * np.median(sw))])
    B_aug = np.concatenate([B, np.zeros((B.shape[0],
                                         E_anchor.size))], axis=1)
    scale = float(np.sqrt(np.sum(np.abs(F_s) ** 2, axis=(1, 2))).max())
    return Phi * np.sqrt(swv)[:, None], B_aug, F_s, qw, fd, scale


def part1_lambda_sweep(lf, E, F, fit, tag):
    A, Y, F_s, qw, fd, scale = weighted_design(
        lf, E, F, np.asarray(fit.e), np.asarray(fit.w))
    G = A.T @ A
    gval, gvec = np.linalg.eigh(G)
    VtY = gvec.T @ (A.T @ Y.T)          # (M, n_ch)
    rows = []
    for lam in np.logspace(-8, 0, 25):
        # X(lambda) = V diag(g/(g+lambda)) V^T A^T Y  (per channel)
        X = gvec @ (gval[:, None] / (gval[:, None] + lam) * VtY)
        C = lf.unpack_coefficients(X, lf.hermitian_channels(
            F_s.shape[1]), F_s.shape[1]) / fd.f_scale
        norms = lf.residue_norms(C)
        model = lf.basis_apply(
            lf.lorentzian_basis(fd.E, fd.scale_centers(
                np.asarray(fit.e)), np.asarray(fit.w) / fd.e_scale), C)
        rms = lf._quad_rms(model, F_s, qw)
        rows.append(dict(lam=float(lam), rel_lam=float(lam / gval.max()),
                         rms=float(rms),
                         cap_ratio=float(norms.max() / scale)))
        log(f"{tag} lam={lam:.1e} (rel {lam/gval.max():.1e}) "
            f"rms {rms:.4e} cap_ratio {norms.max()/scale:.4g}")
    best = [r for r in rows if r["cap_ratio"] <= 10.0]
    pick = min(best, key=lambda r: r["rms"]) if best else None
    log(f"{tag} smallest-acceptable-lambda pick: {pick}")
    return dict(rows=rows, pick=pick)


def band_rms(lf, E, F, fit, bands):
    """Per-band frobenius rms of the fit vs the data (raw grid)."""
    model = fit.evaluate(E)
    out = {}
    for a, b in bands:
        m = (E >= a) & (E < b)
        out[f"{a}_{b}"] = float(lf.frobenius_rms_error(
            model[m], np.asarray(F)[m]))
    return out


def part2_sep_sweep(lf, R_f2, R_ag):
    results = {}

    # ---- Fig2Ag5 lead0, tgt1e4-class config ----
    E, F = lf.extract_dense(R_f2, 0, 0)
    KW = dict(n_init=24, max_poles=100, target_rms=1e-4,
              psd=True, psd_force=False, eta_raw=2e-2,
              eta_floor_mult=1.0, w_max_raw=0.5,
              local_floor_frac=1.5,
              tr_solver="gram", gram_cut=1e-9)
    bands = [(-6, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6),
             (6, 10)]
    for alpha in (None, 0.05, 0.1, 0.2):
        t = time.time()
        r = lf.fit(E, F, **KW, sep_alpha=alpha)
        h = r.history
        row = dict(sep_alpha=alpha, wall=time.time() - t,
                   n_poles=int(r.n_poles),
                   frob=float(lf.frobenius_rms_error(r.evaluate(E), F)),
                   cap_failed=bool(h.get("residue_cap_failed", False)),
                   n_dropped=int(h.get("n_dropped", 0) or 0),
                   n_clipped=int(h.get("n_clipped", 0) or 0),
                   n_sep_merged=int(h.get("n_sep_merged", 0)),
                   cancellation_index=h.get("cancellation_index"),
                   design_kappa=h.get("design_kappa"),
                   bands=band_rms(lf, E, F, r, bands))
        scale = float(np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2))).max())
        row["cap_ratio"] = float(
            lf.residue_norms(np.asarray(r.C)).max() / scale)
        results[f"Fig2Ag5_sep_{alpha}"] = row
        log(f"Fig2Ag5 sep={alpha}: n={row['n_poles']} "
            f"frob {row['frob']:.4e} cap {row['cap_ratio']:.3g} "
            f"failed={row['cap_failed']} "
            f"sep_merged={row['n_sep_merged']} "
            f"band2_4={row['bands']['2_4']:.4e}")

    # ---- Ag3Au3 lead0, m1_w03 config ----
    E, F = lf.extract_dense(R_ag, 0, 0)
    KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
              psd=True, psd_force=False, eta_raw=1e-2,
              eta_floor_mult=1.0, w_max_raw=0.3,
              local_floor_frac=1.5,
              tr_solver="gram", gram_cut=1e-9)
    bands = [(-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 9)]
    for alpha, stab in ((None, 1.0), (0.1, 1.0), (0.2, 1.0),
                        (None, 0.0)):
        t = time.time()
        r = lf.fit(E, F, **KW, sep_alpha=alpha, stab_weight=stab)
        h = r.history
        row = dict(sep_alpha=alpha, stab_weight=stab,
                   wall=time.time() - t,
                   n_poles=int(r.n_poles),
                   frob=float(lf.frobenius_rms_error(r.evaluate(E), F)),
                   cap_failed=bool(h.get("residue_cap_failed", False)),
                   n_dropped=int(h.get("n_dropped", 0) or 0),
                   n_clipped=int(h.get("n_clipped", 0) or 0),
                   n_sep_merged=int(h.get("n_sep_merged", 0)),
                   cancellation_index=h.get("cancellation_index"),
                   design_kappa=h.get("design_kappa"),
                   bands=band_rms(lf, E, F, r, bands))
        scale = float(np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2))).max())
        row["cap_ratio"] = float(
            lf.residue_norms(np.asarray(r.C)).max() / scale)
        key = f"Ag3Au3_sep_{alpha}_stab_{stab}"
        results[key] = row
        log(f"Ag3Au3 sep={alpha} stab={stab}: n={row['n_poles']} "
            f"frob {row['frob']:.4e} cap {row['cap_ratio']:.3g} "
            f"failed={row['cap_failed']} "
            f"canc={row['cancellation_index']:.4g}")
    return results


def main():
    lf, R_f2, R_ag = load_all()
    out = {}
    # Part 1: closed-form lambda sweeps at the two monsters' poles
    E, F = lf.extract_dense(R_f2, 0, 0)
    fit = np.load(ROOT + "/Data2/Fig2Ag5/sweep_3new/"
                  "tgt1e4_lead0_ik0_c0.npz",
                  allow_pickle=True)["fit"].item()
    out["lambda_Fig2Ag5_tgt1e4"] = part1_lambda_sweep(
        lf, E, F, fit, "Fig2Ag5")
    E, F = lf.extract_dense(R_ag, 0, 0)
    fit = np.load(ROOT + "/Data2/Ag3Au3/sweep_ag3au3/"
                  "m1_w03_lead0_ik0_s1.npz",
                  allow_pickle=True)["fit"].item()
    out["lambda_Ag3Au3_m1w03"] = part1_lambda_sweep(
        lf, E, F, fit, "Ag3Au3")
    # Part 2: sep_alpha + stab_weight fits
    out["sep_sweep"] = part2_sep_sweep(lf, R_f2, R_ag)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
