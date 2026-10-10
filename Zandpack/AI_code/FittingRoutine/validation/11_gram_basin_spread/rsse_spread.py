"""Perturbed-start basin spread, VerySmallRSSE lead0 (the second
system; the E=0-notch lead, psd_force=False, cheap fits).

Per run: the production fit_boosted with the compare_dos_t.py kwargs
(n_init=24, mx=150, tgt=5e-4, psd=True, psd_force=False, eta=1e-2,
mult=3, w_max=0.5, boost n_res=8) on the CURRENT 189-pt grid
(refined/C1.xz), seeded via fit(init_poles=...).

T-level: fresh pickle, inject {lead0: candidate, lead1: FIXED partner
refit once on the current grid with the exact solver (the exactera
cache predates the densification - HANDOFF 4.11 warning)},
find_correction (default window, as compare_dos_t.py), Renormalise_H,
Inspect_transmission_from_hilbert_transform over (-5,5) at eta=1e-2,
vs TBtrans (R.tbtT at the sampling indices).

NOTE: there is no trustworthy stored baseline for this grid, so the c0
control here does not gate against a cache - it IS the baseline.
Resume: skips (solver, start) pairs already in rsse_results.jsonl.
OMP_NUM_THREADS=1.
"""
import lzma
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spread_common as sc  # noqa: E402


def load_pkl():
    with lzma.open("C1.xz", "rb") as f:
        return pickle.load(f)


def main():
    os.chdir(os.path.join(sc.ROOT, "Data2", "VerySmallRSSE",
                          "refined"))

    R = load_pkl()                     # pickles FIRST
    lf = sc.import_ai_fit()

    lead, ik = 0, 0
    E, F = lf.extract_dense(R, lead, ik)
    c0, w0, fd = sc.baseline_init_poles(lf, E, F)
    print(f"[rsse] lead{lead}: {E.size} pts, p={F.shape[1]}, "
          f"{c0.size} init poles", flush=True)

    BASE_KW = dict(n_init=24, max_poles=150, target_rms=5e-4,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=3.0)     # p=6 -> channel_tol 0.0

    # FIXED partner: lead1 refit once on the CURRENT grid
    t0 = time.perf_counter()
    E1, F1 = lf.extract_dense(R, 1, 0)
    fixed_lead1 = lf.fit_boosted(E1, F1, dict(BASE_KW,
                                             tr_solver="exact"),
                                 n_res=8)
    print(f"[rsse] fixed lead1 refit: {fixed_lead1.n_poles} poles "
          f"rms {fixed_lead1.rms:.4g} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    # TBtrans reference (from the pristine sampling)
    sidx = R.sampling_idx[0]
    E_exact = np.asarray(R.Contour[sidx]).real
    T_exact = np.asarray(R.tbtT[sidx]).real

    E_line = np.asarray(R.Contour[R.sampling_idx[0]]).real
    starts = sc.make_starts(c0.size, fd.e_scale)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "rsse_results.jsonl")
    done = sc.done_keys(out)

    from Zandpack.FittingTools import find_correction  # noqa: E402

    def t_metrics(cand):
        R2 = load_pkl()
        lf.inject_fits(R2, {(0, 0): cand, (1, 0): fixed_lead1})
        R2.Renormalise_H(find_correction(R2))
        Eg = np.linspace(-5, 5, 300)
        _, T_new = R2.Inspect_transmission_from_hilbert_transform(
            Eg, eta=1e-2, return_result=True)
        T_exact_at = np.interp(Eg, E_exact, T_exact)
        return dict(t_rms=sc.rms(T_new.real, T_exact_at),
                    t_osc=float(np.abs(
                        np.diff(T_new.real - T_exact_at)).mean()))

    for start, de, dw in starts:
        ce = np.asarray(c0, dtype=float) + de
        we = np.asarray(w0, dtype=float) + dw
        for sname, skw in sc.SOLVERS:
            if (sname, start) in done:
                print(f"[rsse] skip {sname} {start} (done)", flush=True)
                continue
            kw = dict(BASE_KW)
            kw.update(skw)
            kw["init_poles"] = (ce, we)
            t0 = time.perf_counter()
            r = lf.fit_boosted(E, F, kw, n_res=8)
            wall = time.perf_counter() - t0
            rec = dict(system="VerySmallRSSE", lead=lead, solver=sname,
                       start=start, wall=wall)
            rec.update(sc.gamma_metrics(r, E_line))
            rec.update(t_metrics(r))
            sc.append_jsonl(out, rec)
            print(f"[rsse] {sname:12s} {start:3s}: "
                  f"wall {wall:6.1f}s rms {rec['rms']:.4g} "
                  f"n={rec['n_poles']} minEig {rec['min_eig']:+.3g} "
                  f"T rms {rec['t_rms']:.4g}", flush=True)
    print("[rsse] DONE", flush=True)


if __name__ == "__main__":
    main()
