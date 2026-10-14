#!/usr/bin/env python3
"""Steps 4-5 experiment (2026-10-13): PSD-add scoring + in-optimizer
ridge, on the two monster-forming cases.

Fig2Ag5 lead0, tgt1e4 config, c0 start (the trajectory whose psd-add
pileup forms the 1.65e8-residue cluster; success = cap-clean WITHOUT
the 13x rms jump, judged on the 2-4 eV band). Ag3Au3 lead0, m1_w03.

Settings:
  baseline     all knobs off (the monster trajectory)
  on30         stab onset 30 (consider + psd-add scored)
  ridge1e-4/1e-3/1e-2   opt_ridge inside every fit_poles stage
  on30+ridge   onset + the best ridge from the sweep
  on30+gain    onset + psd_gain_tol stopping rule

Run:  OMP_NUM_THREADS=1 python validation/step45_experiment.py
Out:  validation/step45_experiment.json
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
OUT = os.path.join(ROOT, "validation", "step45_experiment.json")
T0 = time.time()


def log(msg):
    print(f"[step45 {time.time()-T0:6.0f}s] {msg}", flush=True)


def load(sysdir, pkl):
    _path0 = list(sys.path)
    with gzip.open(os.path.join(ROOT, sysdir, pkl), "rb") as f:
        R = pickle.load(f)
    for q in [q for q in sys.path if q not in _path0]:
        sys.path.remove(q)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
    lf = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base"] = lf
    spec.loader.exec_module(lf)
    from Zandpack.AI_code.FittingRoutine import AI_fit_sweep as sf
    return lf, sf, R


def band_rms(lf, E, F, fit, bands):
    model = fit.evaluate(E)
    return {f"{a}_{b}": float(lf.frobenius_rms_error(
        model[(E >= a) & (E < b)], np.asarray(F)[(E >= a) & (E < b)]))
        for a, b in bands}


def run_case(name, lf, sf, R, lead, kw, bands, settings):
    E, F = lf.extract_dense(R, lead, 0)
    fd = lf.prepare(E, F)
    c0, w0 = lf._poles_init_dispatch(
        "skrf", fd.E, fd.F, kw["n_init"], e_margin=0.15, psd=True)
    scale = float(np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2))).max())
    out = {}
    for sname, extra in settings.items():
        t = time.time()
        r = lf.fit(E, F, **kw, init_poles=(c0, w0), **extra)
        h = r.history
        row = dict(wall=time.time() - t,
                   n_poles=int(r.n_poles),
                   n_psd_added=int(h.get("n_psd_added", 0) or 0),
                   frob=float(lf.frobenius_rms_error(r.evaluate(E), F)),
                   quad=float(h.get("rms", np.nan)),
                   cap_ratio=float(lf.residue_norms(
                       np.asarray(r.C)).max() / scale),
                   cap_failed=bool(h.get("residue_cap_failed", False)),
                   max_cand_ratio=float(h.get("max_cand_ratio", np.nan)),
                   bands=band_rms(lf, E, F, r, bands))
        out[sname] = row
        band_key = "2_4" if "2_4" in row["bands"] else list(
            row["bands"])[2]
        log(f"{name} {sname:<14}: n={row['n_poles']} "
            f"psd_added={row['n_psd_added']} frob {row['frob']:.4e} "
            f"cap {row['cap_ratio']:.3g} "
            f"failed={row['cap_failed']} "
            f"band {band_key}={row['bands'][band_key]:.4e} "
            f"[{row['wall']:.0f}s]")
    return out


def main():
    results = {}
    # ---- Fig2Ag5 tgt1e4 ----
    lf, sf, R = load("Data2/Fig2Ag5", "tmp_refined_deriv.gz")
    kw = dict(n_init=24, max_poles=100, target_rms=1e-4,
              psd=True, psd_force=False, eta_raw=2e-2,
              eta_floor_mult=1.0, w_max_raw=0.5,
              local_floor_frac=1.5,
              tr_solver="gram", gram_cut=1e-9)
    bands = [(-6, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6),
             (6, 10)]
    settings = {
        "baseline": dict(),
        "on30": dict(stab_weight=1.0, stab_onset=30.0),
        "ridge1e-4": dict(opt_ridge=1e-4),
        "ridge1e-3": dict(opt_ridge=1e-3),
        "ridge1e-2": dict(opt_ridge=1e-2),
        "on30+ridge1e-3": dict(stab_weight=1.0, stab_onset=30.0,
                                opt_ridge=1e-3),
        "on30+gain02": dict(stab_weight=1.0, stab_onset=30.0,
                             psd_gain_tol=0.02),
    }
    results["Fig2Ag5_tgt1e4"] = run_case(
        "Fig2Ag5", lf, sf, R, 0, kw, bands, settings)

    # ---- Ag3Au3 m1_w03 ----
    lf, sf, R = load("Data2/Ag3Au3", "tmp_refined_deriv.gz")
    kw = dict(n_init=24, max_poles=100, target_rms=2e-3,
              psd=True, psd_force=False, eta_raw=1e-2,
              eta_floor_mult=1.0, w_max_raw=0.3,
              local_floor_frac=1.5,
              tr_solver="gram", gram_cut=1e-9)
    bands = [(-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 9)]
    settings = {
        "baseline": dict(),
        "on30": dict(stab_weight=1.0, stab_onset=30.0),
        "ridge1e-3": dict(opt_ridge=1e-3),
        "on30+ridge1e-3": dict(stab_weight=1.0, stab_onset=30.0,
                                opt_ridge=1e-3),
        "on30+gain02": dict(stab_weight=1.0, stab_onset=30.0,
                             psd_gain_tol=0.02),
    }
    results["Ag3Au3_m1w03"] = run_case(
        "Ag3Au3", lf, sf, R, 0, kw, bands, settings)

    json.dump(results, open(OUT, "w"), indent=1, default=float)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
