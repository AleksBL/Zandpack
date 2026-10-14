#!/usr/bin/env python3
"""Step 1 (evidence round): bit-identity regression, SELF-CONTROLLED.

For each clean system: one config, the SAME c0 start, one process.
Fit A with the OLD semantics (residue_cap_mult=None, stab_weight=0,
sep_alpha=None) and fit B with the NEW DEFAULTS (cap 10, stab 1.0,
sep None). On clean fits they must be BIT-IDENTICAL - the defaults
must be exact no-ops below the cap ratio.

Run:  OMP_NUM_THREADS=1 python validation/bitidentity_check.py
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
OUT = os.path.join(ROOT, "validation", "bitidentity_check.json")
T0 = time.time()


def log(msg):
    print(f"[bitid {time.time()-T0:6.0f}s] {msg}", flush=True)


def _base(**kw):
    d = dict(n_init=24, psd=True, local_floor_frac=1.5,
             tr_solver="gram", gram_cut=1e-9)
    d.update(kw)
    return d


CASES = {
    "Fig2Ag5": dict(pickle="Data2/Fig2Ag5/tmp_refined_deriv.gz",
                    lead=0, ik=0,
                    base=_base(max_poles=100, target_rms=2e-3,
                               psd_force=False, eta_raw=2e-2,
                               eta_floor_mult=0.5, w_max_raw=0.5),
                    extra={}),
    "nogap_Ag5": dict(pickle="Data2/nogap_Ag5/tmp_refined_deriv.gz",
                      lead=0, ik=0,
                      base=_base(max_poles=90, target_rms=2e-3,
                                 psd_force=False, eta_raw=2e-2,
                                 eta_floor_mult=1.0, w_max_raw=0.3),
                      extra={}),
    "zgnr_spin": dict(pickle="Data2/zgnr_spin/tmp.gz",
                      lead=0, ik=0,
                      base=_base(max_poles=100, target_rms=2e-3,
                                 psd_force=True, eta_raw=2e-2,
                                 eta_floor_mult=5.0, w_max_raw=0.5),
                      extra={}),
    "agnr": dict(pickle="Data2/agnr/tmp_refined_deriv.gz",
                 lead=0, ik=0,
                 base=_base(max_poles=150, target_rms=1e-3,
                             psd_force=False, eta_raw=1e-2,
                             eta_floor_mult=1.0, w_max_raw=0.5),
                 extra=dict(target_rms=1e-4)),
}


def main():
    _path0 = list(sys.path)
    pickles = {}
    for name, spec in CASES.items():
        with gzip.open(os.path.join(ROOT, spec["pickle"]), "rb") as f:
            pickles[name] = pickle.load(f)
    for q in [q for q in sys.path if q not in _path0]:
        sys.path.remove(q)
    import importlib.util
    spec_m = importlib.util.spec_from_file_location(
        "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
    lf = importlib.util.module_from_spec(spec_m)
    sys.modules["AI_fit_base"] = lf
    spec_m.loader.exec_module(lf)
    from Zandpack.AI_code.FittingRoutine import AI_fit_sweep as sf

    results = {}
    for name, spec in CASES.items():
        R = pickles[name]
        E, F = lf.extract_dense(R, spec["lead"], spec["ik"])
        sw = sf.GammaSweep(lf, E, F, cache_dir="/tmp/bitid_cache",
                           base_kw=dict(spec["base"]), n_res=8,
                           n_starts=3, lead_label="bitid",
                           verbose=False)
        starts = dict((t, (c0, w0)) for t, c0, w0
                      in sw._config_starts(spec["extra"]))
        c0, w0 = starts["c0"]
        kw = dict(spec["base"])
        kw.update(spec["extra"])
        tA = time.time()
        fitA = lf.fit(E, F, **kw, init_poles=(c0, w0),
                      residue_cap_mult=None, stab_weight=0.0,
                      sep_alpha=None)
        wallA = time.time() - tA
        tB = time.time()
        fitB = lf.fit(E, F, **kw, init_poles=(c0, w0))
        wallB = time.time() - tB
        same = (np.array_equal(np.asarray(fitA.e),
                               np.asarray(fitB.e))
                and np.array_equal(np.asarray(fitA.w),
                                   np.asarray(fitB.w))
                and np.array_equal(np.asarray(fitA.C),
                                   np.asarray(fitB.C)))
        results[name] = dict(
            n_poles=int(fitA.n_poles),
            bit_identical=bool(same),
            frob=float(lf.frobenius_rms_error(fitA.evaluate(E), F)),
            wallA=wallA, wallB=wallB,
            cap_ratio_B=float(
                lf.residue_norms(np.asarray(fitB.C)).max()
                / np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2))).max()))
        log(f"{name}: n={fitA.n_poles} "
            f"bit-identical(old-semantics vs new-defaults)={same} "
            f"frob {results[name]['frob']:.4e} "
            f"cap_ratio_B {results[name]['cap_ratio_B']:.3g} "
            f"[A {wallA:.0f}s B {wallB:.0f}s]")
    json.dump(results, open(OUT, "w"), indent=1, default=float)
    ok = all(r["bit_identical"] for r in results.values())
    log(f"VERDICT: {'ALL BIT-IDENTICAL - defaults are no-ops' if ok else 'DIFFERENCES FOUND - fix before proceeding'}")


if __name__ == "__main__":
    main()
