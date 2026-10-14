#!/usr/bin/env python3
"""CENSUS of pathology diagnostics over EVERY existing sweep cache.

Per cache (Data2 systems, all eras): per-pole noise amplification
diag((A^T A)^-1) on the WEIGHTED+anchored design, design condition
number kappa(A), cancellation index sum_i max_n ||C_i phi_i(E_n)|| /
max_n ||Gamma(E_n)||, residue cap ratio, dense-grid envelope
max||Gamma||/max||F|| and min eig on window + padded grids + the
Fermi-pole contour points.

Purpose (review 2026-10-13): the evidence so far rests on Fig2Ag5 +
one Ag3Au3 fit. The census gives the DISTRIBUTION of each metric,
tells which fixes matter, and anchors the gate thresholds.

Run:  OMP_NUM_THREADS=1 python validation/fit_census.py
Out:  validation/fit_census.json
"""
import glob
import gzip
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
DATA2 = os.path.join(ROOT, "Data2")
OUT = os.path.join(ROOT, "validation", "fit_census.json")

T0 = time.time()


def log(msg):
    print(f"[census {time.time()-T0:6.0f}s] {msg}", flush=True)


def load_fit_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
    lf = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base"] = lf
    spec.loader.exec_module(lf)
    # all three era module names (see cap_distribution note)
    sys.modules.setdefault("lorentz_fit", lf)
    sys.modules.setdefault("AI_fit", lf)
    return lf


def system_pickle(sysdir):
    for pat in ("tmp_refined_deriv.gz", "tmp_refined*.gz", "tmp.gz"):
        cands = sorted(glob.glob(os.path.join(sysdir, pat)),
                       key=os.path.getmtime)
        if cands:
            return cands[-1]
    return None


def diagnostics(lf, E, F, fit, R=None, lead=None, pad=5.0):
    """All census metrics for one fit on its (E, F) slice."""
    C = np.asarray(fit.C)
    e = np.asarray(fit.e, dtype=float)
    w = np.asarray(fit.w, dtype=float)
    E = np.asarray(E, dtype=float)
    norms = lf.residue_norms(C)
    scale = float(np.sqrt(np.sum(np.abs(np.asarray(F)) ** 2,
                                 axis=(1, 2))).max())

    # ---- weighted + anchored design conditioning ----
    fd = lf.prepare(E, F)
    E_s, F_s = fd.E, fd.F
    chans = lf.hermitian_channels(F_s.shape[1])
    qw = lf._quad_widths(E_s)
    sw = lf._sample_weights(F_s, "relative", quad=qw)
    span = float(E_s.max() - E_s.min())
    d = np.array([0.05, 0.15, 0.35, 0.65]) * 0.5 * span
    E_anchor = np.concatenate([E_s.min() - d, E_s.max() + d])
    E_aug = np.concatenate([E_s, E_anchor])
    Phi = lf.lorentzian_basis(E_aug, fd.scale_centers(e),
                              w / fd.e_scale)
    if sw is None:
        swv = np.ones(E_aug.size)
    else:
        swv = np.concatenate([sw, np.full(E_anchor.size,
                                           0.05 * np.median(sw))])
    Aw = Phi * np.sqrt(swv)[:, None]        # (N_aug, M)
    G = Aw.T @ Aw                            # (M, M)
    gval, gvec = np.linalg.eigh(G)
    amp2 = np.sum(gvec ** 2 / np.maximum(gval, 1e-300), axis=1)
    # amplification of unit noise into pole i's coefficient (the
    # per-pole diagonal of the inverse Gram, sqrt'd)
    amp = np.sqrt(np.maximum(amp2, 0.0))
    kappa = float(np.sqrt(gval.max() / max(gval.min(), 1e-300)))

    # ---- cancellation index (in-window contributions) ----
    Phi_s = lf.lorentzian_basis(E_s, fd.scale_centers(e), w / fd.e_scale)
    # per-pole contribution norms on the samples: (N, M)
    contrib = (np.abs(Phi_s[:, :, None, None] * (C / fd.f_scale)
                      [None, :, :, :]) ** 2).sum(axis=(2, 3))
    gamma_n = np.sqrt(np.sum(np.abs(F_s) ** 2, axis=(1, 2)))
    canc = float(np.sqrt(contrib).max(axis=0).sum()
                 / max(gamma_n.max(), 1e-300))

    # ---- dense-grid envelope + min eig (window + pad) ----
    lo, hi = float(E.min()), float(E.max())
    grid = np.linspace(lo - pad, hi + pad, 1001)
    model = fit.evaluate(grid)
    env = float(np.sqrt(np.sum(np.abs(model) ** 2,
                               axis=(1, 2))).max() / max(scale, 1e-300))
    mev = np.linalg.eigvalsh(0.5 * (model + np.conj(
        np.swapaxes(model, 1, 2))))
    min_eig_dense = float(mev.min())
    fermi_min = None
    if R is not None and lead is not None:
        try:
            z = np.asarray(R.Contour[np.asarray(R.F_poles_idx[lead])])
            Phi_z = (w ** 2)[None, :] / ((z[:, None] - e[None, :]) ** 2
                                         + (w ** 2)[None, :])
            Fz = np.einsum("nk,kij->nij", Phi_z, C)
            evz = np.linalg.eigvalsh(0.5 * (Fz + np.conj(
                np.swapaxes(Fz, 1, 2))))
            fermi_min = float(evz.min())
        except Exception:                                   # noqa: BLE001
            fermi_min = None

    return dict(n_poles=int(fit.n_poles),
                cap_ratio=float(norms.max() / max(scale, 1e-300)),
                kappa=float(kappa),
                max_amplification=float(amp.max()),
                worst_pole=int(np.argmax(amp)),
                cancellation_index=canc,
                envelope=env,
                min_eig_dense=min_eig_dense,
                fermi_min=fermi_min)


def main():
    lf = load_fit_module()
    results = []
    systems = sorted(d for d in glob.glob(os.path.join(DATA2, "*"))
                     if os.path.isdir(d)
                     and glob.glob(os.path.join(d, "sweep*", "*.npz")))
    for sysdir in systems:
        name_sys = os.path.basename(sysdir)
        pkl = system_pickle(sysdir)
        if pkl is None:
            log(f"{name_sys}: NO pickle - skipped")
            continue
        _path0 = list(sys.path)
        with gzip.open(pkl, "rb") as f:
            R = pickle.load(f)
        for q in [q for q in sys.path if q not in _path0]:
            sys.path.remove(q)
        caches = sorted(glob.glob(os.path.join(sysdir, "sweep*",
                                                "*.npz")))
        log(f"{name_sys}: {len(caches)} caches, "
                 f"{os.path.basename(pkl)}")
        for cache in caches:
            base = os.path.basename(cache)[:-4]
            parts = base.split("_")
            try:
                lead = int(parts[-3].replace("lead", ""))
                ik = int(parts[-2].replace("ik", ""))
            except (ValueError, IndexError):
                continue
            try:
                fit = np.load(cache, allow_pickle=True)["fit"].item()
                E, F = lf.extract_dense(R, lead, ik)
                row = dict(system=name_sys, cache=base)
                row.update(diagnostics(lf, E, F, fit, R=R, lead=lead))
                results.append(row)
            except Exception as exc:                       # noqa: BLE001
                results.append(dict(system=name_sys, cache=base,
                                    error=str(exc)[:80]))
        done = [r for r in results if r["system"] == name_sys
                and "error" not in r]
        if done:
            log(f"{name_sys}: {len(done)} ok; median cap_ratio "
                f"{np.median([r['cap_ratio'] for r in done]):.3g}, "
                f"median kappa {np.median([r['kappa'] for r in done]):.3g}, "
                f"median amp "
                f"{np.median([r['max_amplification'] for r in done]):.3g}, "
                f"median canc "
                f"{np.median([r['cancellation_index'] for r in done]):.3g}")
    json.dump(results, open(OUT, "w"), indent=1, default=float)
    log(f"wrote {OUT} ({len(results)} rows)")
    # aggregate
    log("=== CENSUS SUMMARY (medians; p90/max for key metrics) ===")
    ok = [r for r in results if "error" not in r]
    for name_sys in sorted({r["system"] for r in ok}):
        rows = [r for r in ok if r["system"] == name_sys]
        cap = np.array([r["cap_ratio"] for r in rows])
        amp = np.array([r["max_amplification"] for r in rows])
        canc = np.array([r["cancellation_index"] for r in rows])
        env = np.array([r["envelope"] for r in rows])
        eig = np.array([r["min_eig_dense"] for r in rows])
        print(f"{name_sys:<14} n={len(rows):<4} "
              f"cap med/p90/max {np.median(cap):.2g}/"
              f"{np.percentile(cap,90):.2g}/{cap.max():.2g}  "
              f"amp {np.median(amp):.2g}/"
              f"{np.percentile(amp,90):.2g}/{amp.max():.2g}  "
              f"canc {np.median(canc):.2g}/"
              f"{np.percentile(canc,90):.2g}/{canc.max():.2g}  "
              f"env {np.median(env):.2g}/{env.max():.2g}  "
              f"eig min {eig.min():.3g}")


if __name__ == "__main__":
    main()
