#!/usr/bin/env python3
"""Item 2: what can Ag3Au3 achieve under a cap?

cap (residue_cap_mult) in {3, 10, 30, 100, 300, 1000} x 6 seeded
starts, stab onset 30 steering (the round's best for this system),
BOTH leads swept independently (production protocol - the electrodes
differ, NO pole inheritance), health-gated per-lead winner selection,
T-level at matched eta vs the sampled-SE reference. Reports per-start
distributions (median + range) and the winner pair per cap.

Run:  OMP_NUM_THREADS=1 python validation/ag3au3_cap_curve.py
Out:  validation/ag3au3_cap_curve.json
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
OUT = os.path.join(ROOT, "validation", "ag3au3_cap_curve.json")
N_JOBS = min(12, len(os.sched_getaffinity(0))) if hasattr(
    os, "sched_getaffinity") else 1
T0 = time.time()
SEED = 20261013
JITTER_RAW = 0.05
CAPS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
ONSET = 30.0
BASE = dict(n_init=24, psd=True, local_floor_frac=1.5,
            tr_solver="gram", gram_cut=1e-9, max_poles=100,
            target_rms=2e-3, psd_force=False, eta_raw=1e-2,
            eta_floor_mult=1.0, w_max_raw=0.3)


def log(msg):
    print(f"[capcurve {time.time()-T0:6.0f}s] {msg}", flush=True)


_R = None
_LF = None


def _purge():
    for p in [p for p in sys.path
              if os.path.basename(p.rstrip(os.sep)) == "siesta_python"
              and os.path.isfile(os.path.join(p, "siesta_python.py"))]:
        sys.path.remove(p)
    sm = sys.modules.get("siesta_python")
    if sm is not None and not hasattr(sm, "__path__"):
        del sys.modules["siesta_python"]


def _lf():
    global _LF
    if _LF is None:
        _purge()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
        _LF = importlib.util.module_from_spec(spec)
        sys.modules["AI_fit_base"] = _LF
        spec.loader.exec_module(_LF)
    return _LF


def _R_():
    global _R
    if _R is None:
        _purge()
        with gzip.open(os.path.join(
                ROOT, "Data2/Ag3Au3/tmp_refined_deriv.gz"), "rb") as f:
            _R = pickle.load(f)
        _purge()
    return _R


def _starts(lf, E, F):
    fd = lf.prepare(E, F)
    c0, w0 = lf._poles_init_dispatch(
        "skrf", fd.E, fd.F, BASE["n_init"], e_margin=0.15, psd=True)
    rng = np.random.default_rng(SEED)
    starts = [("c0", c0, w0)]
    for s in range(1, 6):
        de = rng.uniform(-JITTER_RAW, JITTER_RAW, c0.size) / fd.e_scale
        dw = rng.uniform(-JITTER_RAW, JITTER_RAW, c0.size) / fd.e_scale
        starts.append((f"s{s}", c0 + de, w0 + dw))
    return starts


def _task(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    lf = _lf()
    R = _R_()
    E, F = lf.extract_dense(R, job["lead"], 0)
    starts = _starts(lf, E, F)
    tag, c0, w0 = starts[job["start_idx"]]
    t = time.time()
    r = lf.fit(E, F, **dict(BASE), init_poles=(c0, w0),
               residue_cap_mult=job["cap"], stab_weight=1.0,
               stab_onset=ONSET)
    h = r.history
    scale = float(np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2))).max())
    Eg = np.linspace(E.min() - 2.0, E.max() + 2.0, 801)
    Fev = r.evaluate(Eg)
    return dict(cap=job["cap"], lead=job["lead"],
                start=job["start_idx"], start_tag=tag,
                wall=time.time() - t, n_poles=int(r.n_poles),
                frob=float(lf.frobenius_rms_error(r.evaluate(E), F)),
                quad_rms=float(h.get("rms", np.nan)),
                res_ratio=float(lf.residue_norms(
                    np.asarray(r.C)).max() / scale),
                cap_failed=bool(h.get("residue_cap_failed", False)),
                min_eig=float(np.linalg.eigvalsh(Fev).min()),
                held_out=float(lf.held_out_error(
                    E, F, r.e, r.w, r.C, test_idx=np.arange(0, E.size, 5))))


def main():
    _path0 = list(sys.path)
    # parent-local module instance: the global _LF must stay None
    # at task-serialization time (a module-valued global is pickled
    # BY REFERENCE and the workers cannot import 'AI_fit_base')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base_local", os.path.join(ROOT, "AI_fit_base.py"))
    lf = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base_local"] = lf
    spec.loader.exec_module(lf)
    R = _R_()
    for q in [q for q in sys.path if q not in _path0]:
        sys.path.remove(q)
    tasks = []
    for cap in CAPS:
        for lead in (0, 1):
            for idx in range(6):
                tasks.append(dict(cap=cap, lead=lead, start_idx=idx))
    log(f"{len(tasks)} fits (6 caps x 2 leads x 6 starts)")
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=2)(
        delayed(_task)(j) for j in tasks)
    log("fits done")

    from Zandpack.AI_code.FittingRoutine import AI_fit_sweep as sf
    E_ref, T_ref, dos_ref = lf.sampled_transmission_dos(R)
    results = {}
    for cap in CAPS:
        fits, meta = {}, {}
        for lead in (0, 1):
            E, F = lf.extract_dense(R, lead, 0)
            starts = _starts(lf, E, F)
            sub = [r for r in recs
                   if r["cap"] == cap and r["lead"] == lead]
            healthy = [r for r in sub
                       if r["quad_rms"] <= 0.05
                       and r["min_eig"] > -1.0
                       and r["res_ratio"] <= 10.0]
            pick = min(healthy or sub, key=lambda r: r["frob"])
            meta[f"lead{lead}"] = dict(
                winner_start=pick["start_tag"],
                winner_healthy=bool(healthy),
                winner=dict((k, pick[k]) for k in
                            ("frob", "quad_rms", "res_ratio",
                             "held_out", "n_poles", "cap_failed")),
                n_healthy=len(healthy),
                per_start=[dict((k, r[k]) for k in
                                ("start_tag", "frob", "quad_rms",
                                 "res_ratio", "held_out", "n_poles",
                                 "min_eig", "cap_failed"))
                           for r in sub])
            # deterministic refit of the winner start for the object
            f = lf.fit(E, F, **dict(BASE),
                       init_poles=(starts[pick["start"]][1],
                                   starts[pick["start"]][2]),
                       residue_cap_mult=cap, stab_weight=1.0,
                       stab_onset=ONSET)
            fits[(lead, 0)] = f
        eq = lf.equalize_pole_counts(fits, mode="dup")
        RX = pickle.load(gzip.open(os.path.join(
            ROOT, "Data2/Ag3Au3/tmp_refined_deriv.gz"), "rb"))
        row = sf.evaluate_pair(
            lf, RX, eq, E_ref, T_ref, dos_ref, teta=1e-2,
            corr=(-5.0, 5.0), twin=(-5.0, 9.0),
            bands=[(-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5),
                   (5, 9)])
        results[f"cap{cap:g}"] = dict(meta, t_rms=float(row["t_rms"]),
                                      dos_rms=float(row["dos_rms"]))
        log(f"cap={cap:g}: l0 {meta['lead0']['winner_start']} "
            f"(healthy {meta['lead0']['n_healthy']}/6, frob "
            f"{meta['lead0']['winner']['frob']:.3e}), l1 "
            f"{meta['lead1']['winner_start']} "
            f"(healthy {meta['lead1']['n_healthy']}/6) -> "
            f"T rms {row['t_rms']:.4g}")
    json.dump(dict(fits=recs, results=results), open(OUT, "w"),
              indent=1, default=float)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
