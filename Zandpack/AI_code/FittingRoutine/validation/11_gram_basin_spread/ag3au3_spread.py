"""Perturbed-start basin spread, Ag3Au3 lead1 (the system pinned to
tr_solver='exact' because lsmr/gram land different adaptive-loop
basins).

Per run: the production fit_boosted with the apply_method.py kwargs
(n_init=24, mx=100, tgt=2e-3, psd=True, psd_force=False, eta=1e-2,
mult=5, w_max=0.5, channel_tol=0.0 at p=27, boost n_res=8), seeded via
fit(init_poles=(jittered psd_dict_init poles)). Control start c0 must
reproduce the exact-era cache (cmp_fits_exactera.npz lead1_ik0,
rms 0.0171 / 94 poles) - the harness validity gate.

Gamma metrics: sample rms, pole count, min eigenvalue on the padded
dense grid (window +-2 eV, 801 pts). T-level: fresh pickle, inject
{lead0: exact-era cached fit (FIXED), lead1: candidate}, find_correction
(-5,5), Renormalise_H, Inspect_transmission_from_hilbert_transform over
(-5,9) at eta=1e-2, vs the sampled-SE reference.

Resume: (solver, start) pairs already in ag3au3_results.jsonl are
skipped. OMP_NUM_THREADS=1. Run from anywhere:
  OMP_NUM_THREADS=1 /home/aleks/oak/bin/python3 ag3au3_spread.py
"""
import gzip
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spread_common as sc  # noqa: E402


def main():
    os.chdir(os.path.join(sc.ROOT, "Data2", "Ag3Au3"))

    # pickles FIRST (Zandpack shadow rule)
    with gzip.open("tmp.gz", "rb") as f:
        R = pickle.load(f)
    lf = sc.import_ai_fit()
    # the cached CombinedFit objects pickle as module 'refine_sampling'
    # (the historical shim) - make it importable BEFORE np.load with
    # allow_pickle reads them (same discipline as the Data2 drivers)
    sys.path.insert(0, sc.ROOT)
    import refine_sampling  # noqa: F401,E402
    sys.path.remove(sc.ROOT)

    lead, ik = 1, 0
    E, F = lf.extract_dense(R, lead, ik)
    c0, w0, fd = sc.baseline_init_poles(lf, E, F)
    print(f"[ag3au3] lead{lead}: {E.size} pts, p={F.shape[1]}, "
          f"{c0.size} init poles (psd_dict_init, scaled units)",
          flush=True)

    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0)      # p=27 -> channel_tol 0.0

    # reference + fixed partner (from the exact-era cache, same grid)
    E_ref, T_ref, _ = lf.sampled_transmission_dos(R)
    d = np.load("cmp_fits_exactera.npz", allow_pickle=True)
    fixed = {(0, 0): d["lead0_ik0"].item()}
    base_lead1 = d["lead1_ik0"].item()

    E_line = np.asarray(R.Contour[R.sampling_idx[0]]).real
    starts = sc.make_starts(c0.size, fd.e_scale)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ag3au3_results.jsonl")
    done = sc.done_keys(out)

    from Zandpack.FittingTools import find_correction  # noqa: E402

    def t_metrics(cand):
        with gzip.open("tmp.gz", "rb") as f:
            R2 = pickle.load(f)
        lf.inject_fits(R2, {**fixed, (lead, ik): cand})
        R2.Renormalise_H(find_correction(R2, Emin=-5, Emax=5.0))
        Eg = np.linspace(-5, 9, 600)
        _, T_new = R2.Inspect_transmission_from_hilbert_transform(
            Eg, eta=1e-2, return_result=True)
        T_ref_at = np.interp(Eg, E_ref, T_ref)
        return dict(t_rms=sc.rms(T_new.real, T_ref_at),
                    t_osc=float(np.abs(
                        np.diff(T_new.real - T_ref_at)).mean()))

    for start, de, dw in starts:
        ce = np.asarray(c0, dtype=float) + de
        we = np.asarray(w0, dtype=float) + dw
        for sname, skw in sc.SOLVERS:
            if (sname, start) in done:
                print(f"[ag3au3] skip {sname} {start} (done)", flush=True)
                continue
            kw = dict(BASE_KW)
            kw.update(skw)
            kw["init_poles"] = (ce, we)
            t0 = time.perf_counter()
            r = lf.fit_boosted(E, F, kw, n_res=8)
            wall = time.perf_counter() - t0
            rec = dict(system="Ag3Au3", lead=lead, solver=sname,
                       start=start, wall=wall)
            rec.update(sc.gamma_metrics(r, E_line))
            rec.update(t_metrics(r))
            sc.append_jsonl(out, rec)
            if start == "c0" and sname == "exact":
                ok = (rec["rms"] == float(base_lead1.rms)
                      and rec["n_poles"] == int(base_lead1.n_poles))
                print(f"[ag3au3] CONTROL exact vs exactera cache: "
                      f"{'REPRODUCED' if ok else 'MISMATCH'} "
                      f"(rms {rec['rms']:.6g} vs {base_lead1.rms:.6g}, "
                      f"n {rec['n_poles']} vs {base_lead1.n_poles})",
                      flush=True)
            print(f"[ag3au3] {sname:12s} {start:3s}: "
                  f"wall {wall:6.1f}s rms {rec['rms']:.4g} "
                  f"n={rec['n_poles']} minEig {rec['min_eig']:+.3g} "
                  f"T rms {rec['t_rms']:.4g}", flush=True)
    print("[ag3au3] DONE", flush=True)


if __name__ == "__main__":
    main()
