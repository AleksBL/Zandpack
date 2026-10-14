#!/usr/bin/env python3
"""Step 2+3 (evidence round): multi-start stability-penalty spread.

For each system (Ag3Au3 pathological, spinpol_ag5 + zgnr minority
pathologies, nogap_Ag5 clean control): 6 seeded perturbed starts
(spread_common pattern), stab_weight in {0, 0.3, 1, 3}, lead0 fit
with the system's tuned config. Per fit: frob, quad rms, cap ratio,
cancellation index, padded envelope + min eig, n_poles, wall,
held-out error. Per (system, weight): health-gated winner -> T-level
via evaluate_pair (lead1 inherits lead0's poles - fixed protocol).

Output: validation/stab_spread.json. Choose the stab_weight default
from the DISTRIBUTION, not from one run.

Run:  OMP_NUM_THREADS=1 python validation/stab_spread.py
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
OUT = os.path.join(ROOT, "validation", "stab_spread.json")
N_JOBS = min(12, len(os.sched_getaffinity(0))) if hasattr(
    os, "sched_getaffinity") else 1
T0 = time.time()
SEED = 20261013
JITTER_RAW = 0.05


def log(msg):
    print(f"[stabsp {time.time()-T0:6.0f}s] {msg}", flush=True)


def _base(**kw):
    d = dict(n_init=24, psd=True, local_floor_frac=1.5,
             tr_solver="gram", gram_cut=1e-9)
    d.update(kw)
    return d


SYSTEMS = {
    "Ag3Au3": dict(pickle="Data2/Ag3Au3/tmp_refined_deriv.gz",
                   base=_base(max_poles=100, target_rms=2e-3,
                              psd_force=False, eta_raw=1e-2,
                              eta_floor_mult=1.0, w_max_raw=0.3),
                   eta=1e-2, teta=1e-2, corr=(-5.0, 5.0),
                   twin=(-5.0, 9.0),
                   bands=[(-5, -3), (-3, -1), (-1, 1), (1, 3),
                          (3, 5), (5, 9)]),
    "spinpol_ag5": dict(pickle="Data2/spinpol_ag5/tmp.gz",
                        base=_base(max_poles=100, target_rms=2e-3,
                                   psd_force=False, eta_raw=2e-2,
                                   eta_floor_mult=1.0, w_max_raw=0.3),
                        eta=2e-2, teta=2e-2, corr=(-2.0, 2.0),
                        twin=(-7.0, 7.0),
                        bands=[(-7, -5), (-5, -3), (-3, -1), (-1, 1),
                               (1, 3), (3, 5), (5, 7)]),
    "zgnr": dict(pickle="Data2/zgnr/tmp_refined_deriv.gz",
                 base=_base(max_poles=100, target_rms=2e-3,
                            psd_force=True, eta_raw=1e-2,
                            eta_floor_mult=1.0, w_max_raw=0.3),
                 eta=1e-2, teta=1e-2, corr=(-3.0, 3.0),
                 twin=(-3.0, 7.0),
                 bands=[(-3, -1), (-1, 1), (1, 3), (3, 5), (5, 7)]),
    "nogap_Ag5": dict(pickle="Data2/nogap_Ag5/tmp_refined_deriv.gz",
                      base=_base(max_poles=90, target_rms=2e-3,
                                 psd_force=False, eta_raw=2e-2,
                                 eta_floor_mult=1.0, w_max_raw=0.3),
                      eta=2e-2, teta=2e-2, corr=(-3.0, 3.0),
                      twin=(-5.0, 9.0),
                      bands=[(-5, -3), (-3, -1), (-1, 1), (1, 3),
                             (3, 5), (5, 9)]),
}
WEIGHTS = (0.0, 0.3, 1.0, 3.0)

_R_CACHE = {}
_LF = None


def _purge_siesta_shadow():
    for p in [p for p in sys.path
              if os.path.basename(p.rstrip(os.sep)) == "siesta_python"
              and os.path.isfile(os.path.join(p, "siesta_python.py"))]:
        sys.path.remove(p)
    sm = sys.modules.get("siesta_python")
    if sm is not None and not hasattr(sm, "__path__"):
        del sys.modules["siesta_python"]


def _get_lf():
    global _LF
    if _LF is None:
        _purge_siesta_shadow()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
        _LF = importlib.util.module_from_spec(spec)
        sys.modules["AI_fit_base"] = _LF
        spec.loader.exec_module(_LF)
    return _LF


def _load_lf_local():
    """Parent-side loader under a DIFFERENT module name: the parent's
    instance must NOT end up in the globals cloudpickle serializes
    with _fit_task (a module-valued _LF is pickled BY REFERENCE and
    the workers cannot import 'AI_fit_base' - the first run died with
    ModuleNotFoundError in the worker unpickle)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base_local", os.path.join(ROOT, "AI_fit_base.py"))
    lf2 = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base_local"] = lf2
    spec.loader.exec_module(lf2)
    return lf2


def _get_R(sysname):
    if sysname not in _R_CACHE:
        _purge_siesta_shadow()
        with gzip.open(os.path.join(
                ROOT, SYSTEMS[sysname]["pickle"]), "rb") as f:
            _R_CACHE[sysname] = pickle.load(f)
        for p in [p for p in sys.path
                  if os.path.basename(p.rstrip(os.sep))
                  == "siesta_python" and os.path.isfile(
                      os.path.join(p, "siesta_python.py"))]:
            sys.path.remove(p)
    return _R_CACHE[sysname]


def _fit_task(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    lf = _get_lf()
    R = _get_R(job["system"])
    E, F = lf.extract_dense(R, 0, 0)
    if "c0" not in job or job["c0"] is None:
        # first worker call computes the start set; parent re-runs
        # the same seeded generator, so all workers agree
        fd = lf.prepare(E, F)
        c0, w0 = lf._poles_init_dispatch(
            "skrf", fd.E, fd.F, job["base"]["n_init"], e_margin=0.15,
            psd=True)
        rng = np.random.default_rng(SEED)
        starts = [("c0", c0, w0)]
        for s in range(1, 6):
            de = rng.uniform(-JITTER_RAW, JITTER_RAW,
                             c0.size) / fd.e_scale
            dw = rng.uniform(-JITTER_RAW, JITTER_RAW,
                             c0.size) / fd.e_scale
            starts.append((f"s{s}", c0 + de, w0 + dw))
        job = dict(job)
        job["c0"], job["w0"] = starts[job["start_idx"]][1], \
            starts[job["start_idx"]][2]
    kw = dict(job["base"])
    t = time.time()
    r = lf.fit(E, F, **kw, init_poles=(job["c0"], job["w0"]),
               stab_weight=job["stab_weight"])
    wall = time.time() - t
    h = r.history
    scale = float(np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2))).max())
    # health metrics (padded, like fit_health)
    Eg = np.linspace(E.min() - 2.0, E.max() + 2.0, 801)
    Fev = r.evaluate(Eg)
    env = float(np.sqrt(np.sum(np.abs(Fev) ** 2,
                               axis=(1, 2))).max() / scale)
    min_eig = float(np.linalg.eigvalsh(Fev).min())
    n = E.size
    ho = float(lf.held_out_error(E, F, r.e, r.w, r.C,
                                 test_idx=np.arange(0, n, 5)))
    return dict(system=job["system"], stab=job["stab_weight"],
                start=job["start_idx"], wall=wall,
                n_poles=int(r.n_poles),
                frob=float(lf.frobenius_rms_error(r.evaluate(E), F)),
                quad_rms=float(h.get("rms", np.nan)),
                cap_ratio=float(lf.residue_norms(
                    np.asarray(r.C)).max() / scale),
                cap_failed=bool(h.get("residue_cap_failed", False)),
                canc=float(h.get("cancellation_index", np.nan)),
                envelope=env, min_eig=min_eig, held_out=ho)


def main():
    _path0 = list(sys.path)
    lf = _load_lf_local()          # parent-local; _LF stays None
    tasks = []
    starts_cache = {}
    for sysname, spec in SYSTEMS.items():
        R = _get_R(sysname)
        E, F = lf.extract_dense(R, 0, 0)
        fd = lf.prepare(E, F)
        c0, w0 = lf._poles_init_dispatch(
            "skrf", fd.E, fd.F, spec["base"]["n_init"], e_margin=0.15,
            psd=True)
        rng = np.random.default_rng(SEED)
        starts = [("c0", c0, w0)]
        for s in range(1, 6):
            de = rng.uniform(-JITTER_RAW, JITTER_RAW,
                             c0.size) / fd.e_scale
            dw = rng.uniform(-JITTER_RAW, JITTER_RAW,
                             c0.size) / fd.e_scale
            starts.append((f"s{s}", c0 + de, w0 + dw))
        starts_cache[sysname] = starts
        for w in WEIGHTS:
            for idx, (tag, c0s, w0s) in enumerate(starts):
                tasks.append(dict(system=sysname, stab_weight=w,
                                  base=dict(spec["base"]),
                                  start_idx=idx, start_tag=tag,
                                  c0=c0s, w0=w0s))
    log(f"{len(tasks)} fits "
        f"({len(SYSTEMS)} systems x {len(WEIGHTS)} weights x 6 starts)")
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=2)(
        delayed(_fit_task)(j) for j in tasks)
    for rec in recs:
        rec["start_tag"] = ["c0", "s1", "s2", "s3", "s4", "s5"][
            rec["start"]]
    log(f"fits done")

    # ---- winner selection per (system, weight) + T level ----
    from Zandpack.AI_code.FittingRoutine import AI_fit_sweep as sf
    results = {}
    for sysname, spec in SYSTEMS.items():
        R = _get_R(sysname)
        E, F = lf.extract_dense(R, 0, 0)
        E_ref, T_ref, dos_ref = lf.sampled_transmission_dos(R)
        rows = [r for r in recs if r["system"] == sysname]
        for w in WEIGHTS:
            sub = [r for r in rows if r["stab"] == w]
            # health gate: quad rms <= 0.05 AND min_eig > -1 AND
            # cap ratio <= 10; among healthy pick best frob
            healthy = [r for r in sub if r["quad_rms"] <= 0.05
                       and r["min_eig"] > -1.0
                       and r["cap_ratio"] <= 10.0]
            pick = min(healthy or sub, key=lambda r: r["frob"])
            # T level: refit the winner start (deterministic) for the
            # object, inherit lead1, evaluate
            starts = starts_cache[sysname]
            t = time.time()
            n_ik = len(R.Nonortho_Gammas[0].vals[0])
            fits = {}
            for ik in range(n_ik):
                E0k, F0k = lf.extract_dense(R, 0, ik)
                f0k = lf.fit(E0k, F0k, **spec["base"],
                             init_poles=(starts[pick["start"]][1],
                                         starts[pick["start"]][2]),
                             stab_weight=w)
                E1k, F1k = lf.extract_dense(R, 1, ik)
                f1k = sf.pole_inherit_fit(
                    lf, E1k, F1k, f0k, psd=spec["base"]["psd"],
                    psd_force=spec["base"]["psd_force"])
                fits[(0, ik)] = f0k
                fits[(1, ik)] = f1k
            eq = lf.equalize_pole_counts(fits, mode="dup")
            RX = pickle.load(gzip.open(os.path.join(
                ROOT, spec["pickle"]), "rb"))
            row = sf.evaluate_pair(lf, RX, eq, E_ref, T_ref, dos_ref,
                                  teta=spec["teta"], corr=spec["corr"],
                                  twin=spec["twin"],
                                  bands=spec["bands"])
            results[f"{sysname}_w{w}"] = dict(
                winner_start=pick["start_tag"],
                winner=dict(frob=pick["frob"],
                            quad_rms=pick["quad_rms"],
                            cap_ratio=pick["cap_ratio"],
                            canc=pick["canc"],
                            envelope=pick["envelope"],
                            held_out=pick["held_out"],
                            n_poles=pick["n_poles"]),
                n_healthy=len(healthy),
                t_rms=float(row["t_rms"]),
                dos_rms=float(row["dos_rms"]),
                t_wall=time.time() - t,
                all_starts=[dict((k, r[k]) for k in
                                 ("start_tag", "frob", "quad_rms",
                                  "cap_ratio", "canc", "envelope",
                                  "min_eig", "held_out", "n_poles",
                                  "wall", "cap_failed"))
                            for r in sub])
            log(f"{sysname} stab={w}: winner {pick['start_tag']} "
                f"frob {pick['frob']:.4e} cap "
                f"{pick['cap_ratio']:.3g} healthy "
                f"{len(healthy)}/6 -> T rms {row['t_rms']:.4g}")
    json.dump(dict(fits=recs, winners=results), open(OUT, "w"),
              indent=1, default=float)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
