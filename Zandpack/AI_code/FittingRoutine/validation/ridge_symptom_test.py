#!/usr/bin/env python3
"""Item 6: the ORIGINAL SYMPTOM test on the ridge fits.

Refit the two monster cases with opt_ridge=1e-3 (the production
candidate), then check what Zandpack's per-pole eigenvalue print
(get_propagation_quantities) would report: eigenvalues of each
residue matrix C_i (that IS what Gl_eig contains - HANDOFF 4.16
item 11). Acceptance: NO +-1e8 eigenvalues; everything at O(data
scale). Also: the assembled Gamma at the Fermi-pole contour, and
the T level for the Fig2Ag5 tgt1e4 ridge pair (item 3's missing
number). Finally one full Zandpack get_propagation_quantities run
to see the real print.

Run:  OMP_NUM_THREADS=1 python validation/ridge_symptom_test.py
Out:  validation/ridge_symptom_test.json
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
OUT = os.path.join(ROOT, "validation", "ridge_symptom_test.json")
T0 = time.time()


def log(msg):
    print(f"[symptom {time.time()-T0:6.0f}s] {msg}", flush=True)


def load_lf():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
    lf = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base"] = lf
    spec.loader.exec_module(lf)
    return lf


def main():
    _path0 = list(sys.path)
    R_f2 = pickle.load(gzip.open(
        ROOT + "/Data2/Fig2Ag5/tmp_refined_deriv.gz", "rb"))
    R_ag = pickle.load(gzip.open(
        ROOT + "/Data2/Ag3Au3/tmp_refined_deriv.gz", "rb"))
    for q in [q for q in sys.path if q not in _path0]:
        sys.path.remove(q)
    lf = load_lf()
    from Zandpack.AI_code.FittingRoutine import AI_fit_sweep as sf

    results = {}

    for name, R, kw, lead in (
            ("Fig2Ag5_tgt1e4", R_f2,
             dict(n_init=24, max_poles=100, target_rms=1e-4,
                  psd=True, psd_force=False, eta_raw=2e-2,
                  eta_floor_mult=1.0, w_max_raw=0.5,
                  local_floor_frac=1.5, tr_solver="gram",
                  gram_cut=1e-9), 0),
            ("Ag3Au3_m1w03", R_ag,
             dict(n_init=24, max_poles=100, target_rms=2e-3,
                  psd=True, psd_force=False, eta_raw=1e-2,
                  eta_floor_mult=1.0, w_max_raw=0.3,
                  local_floor_frac=1.5, tr_solver="gram",
                  gram_cut=1e-9), 0)):
        E, F = lf.extract_dense(R, lead, 0)
        fd = lf.prepare(E, F)
        c0, w0 = lf._poles_init_dispatch(
            "skrf", fd.E, fd.F, kw["n_init"], e_margin=0.15,
            psd=True)
        t = time.time()
        r = lf.fit(E, F, **kw, init_poles=(c0, w0), opt_ridge=1e-3)
        h = r.history
        scale = float(np.sqrt(np.sum(np.abs(F) ** 2,
                                    axis=(1, 2))).max())
        C = np.asarray(r.C)
        lam_c = np.linalg.eigh(0.5 * (C + np.conj(
            np.swapaxes(C, 1, 2))))[0]
        z = np.asarray(R.Contour[np.asarray(R.F_poles_idx[lead])])
        Phi_z = (np.asarray(r.w) ** 2)[None, :] / (
            (z[:, None] - np.asarray(r.e)[None, :]) ** 2
            + (np.asarray(r.w) ** 2)[None, :])
        Fz = np.einsum("nk,kij->nij", Phi_z, C)
        evz = np.linalg.eigvalsh(0.5 * (Fz + np.conj(
            np.swapaxes(Fz, 1, 2))))
        results[name] = dict(
            wall=time.time() - t, n_poles=int(r.n_poles),
            frob=float(lf.frobenius_rms_error(r.evaluate(E), F)),
            data_scale=scale,
            perpole_eig_max=float(lam_c.max()),
            perpole_eig_min=float(lam_c.min()),
            res_over_data=float(lf.residue_norms(C).max() / scale),
            ridge_used=float(h.get("ridge_used", np.nan)),
            n_psd_added=int(h.get("n_psd_added", 0) or 0),
            fermi_min=float(evz.min()), fermi_max=float(evz.max()))
        log(f"{name}: n={r.n_poles} frob "
            f"{results[name]['frob']:.4e} "
            f"per-pole eig max {lam_c.max():+.4g} "
            f"(data scale {scale:.4g}, NO 1e8 pathology "
            f"{'CONFIRMED' if abs(lam_c.max()) < 100 * scale else 'FAILED'}); "
            f"residues/data {results[name]['res_over_data']:.3g} "
            f"fermi [{evz.min():+.3g},{evz.max():+.3g}] "
            f"ridge_used {h.get('ridge_used')}")

    # ---- T level for the Fig2Ag5 tgt1e4 ridge pair (item 3) ----
    E, F = lf.extract_dense(R_f2, 0, 0)
    fd = lf.prepare(E, F)
    c0, w0 = lf._poles_init_dispatch(
        "skrf", fd.E, fd.F, 24, e_margin=0.15, psd=True)
    f0 = lf.fit(E, F, n_init=24, max_poles=100, target_rms=1e-4,
                psd=True, psd_force=False, eta_raw=2e-2,
                eta_floor_mult=1.0, w_max_raw=0.5,
                local_floor_frac=1.5, tr_solver="gram",
                gram_cut=1e-9, init_poles=(c0, w0), opt_ridge=1e-3)
    E1, F1 = lf.extract_dense(R_f2, 1, 0)
    f1 = sf.pole_inherit_fit(lf, E1, F1, f0, psd=True,
                             psd_force=False)
    eq = lf.equalize_pole_counts({(0, 0): f0, (1, 0): f1},
                                 mode="dup")
    E_ref, T_ref, dos_ref = lf.sampled_transmission_dos(R_f2)
    RX = pickle.load(gzip.open(
        ROOT + "/Data2/Fig2Ag5/tmp_refined_deriv.gz", "rb"))
    row = sf.evaluate_pair(
        lf, RX, eq, E_ref, T_ref, dos_ref, teta=2e-2,
        corr=(-3.0, 3.0), twin=(-6.0, 10.0),
        bands=[(-6, -4), (-4, -2), (-2, 0), (0, 2), (2, 4),
               (4, 6), (6, 10)])
    results["Fig2Ag5_tgt1e4_T"] = dict(
        t_rms=float(row["t_rms"]), dos_rms=float(row["dos_rms"]),
        n_eq=int(eq[(0, 0)].n_poles),
        band_t=row.get("band_t", {}))
    log(f"Fig2Ag5 tgt1e4 ridge pair: T rms {row['t_rms']:.4g} "
        f"DOS {row['dos_rms']:.4g} n_eq {eq[(0, 0)].n_poles} "
        f"(references: uncapped monster 0.006682, ef05 winner "
        f"0.006196)")

    # ---- one full Zandpack print (the real symptom) ----
    try:
        RX2 = pickle.load(gzip.open(
            ROOT + "/Data2/Fig2Ag5/tmp_refined_deriv.gz", "rb"))
        lf.inject_fits(RX2, eq)
        from Zandpack.FittingTools import find_correction
        RX2.Renormalise_H(find_correction(RX2, Emin=-3.0, Emax=3.0))
        RX2.get_propagation_quantities()
        log("full get_propagation_quantities ran - see the printed "
            "Max/Min eigenvalues above")
    except Exception as exc:                                # noqa: BLE001
        log(f"full Zandpack print failed: {exc}")

    json.dump(results, open(OUT, "w"), indent=1, default=float)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
