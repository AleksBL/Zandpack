"""Production test of the gram solver (tr_solver='gram', gram_cut=1e-9)
on Ag3Au3 - the system historically pinned to 'exact' - using the
recommended recipe from the basin-spread round: 3 starts per lead
(control + 2 jittered, the spread_common seeds), health-gate selection
(padded min eig > -1 AND sample rms <= 0.05, best rms among healthy),
then the full Zandpack post-processing (inject + find_correction +
Renormalise_H + Inspect_transmission_from_hilbert_transform) vs the
sampled-SE reference.

References for the final T rms over (-5, 9) at eta=1e-2:
  old 3.py pipeline (stored)      0.0383
  exactera cached pair (stored)   0.0100  (measured 0.009958)
  exact, single control start     3.44    (spread round, catastrophic)
  gram@1e-9 single-start spread   0.0097-0.196 across starts

Also reports walls: 6 gram fits total (~15 min) vs the ~45-70 min the
same 6 fits would take with 'exact'.

OMP_NUM_THREADS=1.
"""
import gzip
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spread_common as sc  # noqa: E402


def main():
    os.chdir("/tmp")
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R = pickle.load(f)
    lf = sc.import_ai_fit()

    # reference from the pristine sampling (before any injection)
    E_ref, T_ref, _ = lf.sampled_transmission_dos(R)
    E_line = np.asarray(R.Contour[R.sampling_idx[0]]).real

    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0,      # p=27 -> channel_tol 0.0
                   tr_solver="gram", gram_cut=1e-9)

    def health(rec):
        """The joint health gate (min eig AND quality together)."""
        return rec["min_eig"] > -1.0 and rec["rms"] <= 0.05

    fits, recs = {}, {}
    for lead in (0, 1):
        E, F = lf.extract_dense(R, lead, 0)
        c0, w0, fd = sc.baseline_init_poles(lf, E, F)
        starts = sc.make_starts(c0.size, fd.e_scale)[:3]  # c0, s1, s2
        for start, de, dw in starts:
            kw = dict(BASE_KW)
            kw["init_poles"] = (np.asarray(c0, float) + de,
                                np.asarray(w0, float) + dw)
            t0 = time.perf_counter()
            r = lf.fit_boosted(E, F, kw, n_res=8)
            wall = time.perf_counter() - t0
            rec = dict(solver="gram_cut1e-9", start=start,
                       lead=lead, wall=wall)
            rec.update(sc.gamma_metrics(r, E_line))
            recs[(lead, start)] = rec
            fits[(lead, start)] = r
            print(f"[prod] lead{lead} {start:3s}: wall {wall:6.1f}s "
                  f"rms {rec['rms']:.4g} n={rec['n_poles']} "
                  f"minEig {rec['min_eig']:+.3g} "
                  f"healthy {health(rec)}", flush=True)
        # health-gate selection per lead
        healthy = [(k, v) for k, v in recs.items()
                   if k[0] == lead and health(v)]
        pool = healthy if healthy else [(k, v) for k, v in recs.items()
                                        if k[0] == lead]
        pick = min(pool, key=lambda kv: kv[1]["rms"])[0]
        print(f"[prod] lead{lead}: selected start {pick[1]} "
              f"({'healthy' if healthy else 'NO HEALTHY START - best of bad lot'})",
              flush=True)
        fits[("pick", lead)] = fits.pop(pick)

    # T-level evaluation of the selected pair
    from Zandpack.FittingTools import find_correction  # noqa: E402
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R2 = pickle.load(f)
    pair = {(0, 0): fits[("pick", 0)], (1, 0): fits[("pick", 1)]}
    lf.inject_fits(R2, pair)
    R2.Renormalise_H(find_correction(R2, Emin=-5, Emax=5.0))
    Eg = np.linspace(-5, 9, 600)
    _, T_new = R2.Inspect_transmission_from_hilbert_transform(
        Eg, eta=1e-2, return_result=True)
    T_ref_at = np.interp(Eg, E_ref, T_ref)
    t_rms = sc.rms(T_new.real, T_ref_at)
    t_osc = float(np.abs(np.diff(T_new.real - T_ref_at)).mean())
    me = min(float(np.linalg.eigvalsh(pair[(l, 0)].evaluate(
        np.linspace(E_line.min() - 2, E_line.max() + 2, 801))).min())
        for l in (0, 1))

    total_fit_wall = sum(v["wall"] for k, v in recs.items())
    print(f"\n===== gram production test, Ag3Au3 =====", flush=True)
    print(f"  T rms vs sampled-SE ref [(-5,9)]: {t_rms:.4g}  "
          f"(osc {t_osc:.4g})", flush=True)
    print(f"  pair padded min eig: {me:+.3g}", flush=True)
    print(f"  poles: l0 {pair[(0,0)].n_poles}, "
          f"l1 {pair[(1,0)].n_poles}", flush=True)
    print(f"  references: old 3.py 0.0383 | exactera pair 0.0100 | "
          f"exact single control start 3.44", flush=True)
    print(f"  total fit wall (6 gram fits): {total_fit_wall:.0f}s "
          f"(~6x that with 'exact')", flush=True)

    out = dict(system="Ag3Au3", solver="gram_cut1e-9",
               t_rms=t_rms, t_osc=t_osc, min_eig=me,
               poles=[int(pair[(0, 0)].n_poles),
                      int(pair[(1, 0)].n_poles)],
               total_fit_wall=total_fit_wall,
               candidates=[{str(k): v for k, v in
                            dict(recs=[recs[k]], **recs[k]).items()}
                           for k in recs])
    with open(os.path.join(HERE, "ag3au3_gram_production.json"),
              "w") as f:
        json.dump({**{f"{l}_{s}": recs[(l, s)] for (l, s) in recs},
                   "result": {k: out[k] for k in
                              ("t_rms", "t_osc", "min_eig", "poles",
                               "total_fit_wall")}}, f, indent=1,
                  default=float)
    np.savez(os.path.join(HERE, "ag3au3_gram_production_fits.npz"),
             **{f"lead{l}_pick": pair[(l, 0)] for l in (0, 1)},
             **{f"lead{l}_{s}": fits[(l, s)]
                for l in (0, 1) for s in ("c0", "s1", "s2")
                if (l, s) in fits})
    print("[prod] DONE -> ag3au3_gram_production.json, "
          "ag3au3_gram_production_fits.npz", flush=True)


if __name__ == "__main__":
    main()
