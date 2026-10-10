"""Where does the time go in a production gram fit? (tr_solver='gram',
gram_cut=1e-9, Ag3Au3 lead1, control start, full fit_boosted with the
apply_method kwargs - the representative calculation of the spread
round.)

cProfile over the whole fit_boosted, then a targeted aggregation:
- the optimizer portion (_trf_gram loop: eigh, select_step, and the
  fun/gram callbacks it invokes - varproj core QR/solves, jac pieces,
  gram_and_grad),
- the per-fit_poles final coefficient solves,
- the PSD machinery (psd_dict_init/nnls, psd_refine, ADMM,
  project_psd, refit_scales),
- the adaptive-loop machinery around fit_poles (prune, insertion,
  merge, basis/rms evaluation),
- one-time costs (prepare, compress_channels).

Reference wall WITHOUT profiler (spread round, same start/solver):
~100-116 s. cProfile inflates high-call-count Python paths; treat
percentages, not absolute walls. OMP_NUM_THREADS=1.
"""
import cProfile
import gzip
import os
import pstats
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, os.path.join(ROOT, "validation",
                                "11_gram_basin_spread"))
import spread_common as sc  # noqa: E402


def main():
    os.chdir("/tmp")
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R = pickle.load(f)
    lf = sc.import_ai_fit()

    E, F = lf.extract_dense(R, 1, 0)
    c0, w0, _ = sc.baseline_init_poles(lf, E, F)
    BASE_KW = dict(n_init=24, max_poles=100, target_rms=2e-3,
                   psd=True, psd_force=False, eta_raw=1e-2,
                   local_floor_frac=1.5, w_max_raw=0.5,
                   eta_floor_mult=5.0,
                   tr_solver="gram", gram_cut=1e-9,
                   init_poles=(c0, w0))

    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    r = lf.fit_boosted(E, F, BASE_KW, n_res=8)
    pr.disable()
    wall = time.perf_counter() - t0
    prof_path = os.path.join(HERE, "ag3au3_gram_c0.prof")
    pr.dump_stats(prof_path)
    print(f"[prof] fit_boosted wall UNDER PROFILER: {wall:.0f}s "
          f"(unprofiled reference ~100-116 s) | result rms {r.rms:.4g} "
          f"n={r.n_poles} (must match the spread c0 record: "
          f"0.01977/92)", flush=True)

    # ---- aggregation --------------------------------------------------
    total = pstats.Stats(pr).total_tt
    by_name = {}
    for (fn, _ln, name), (cc, nc, tt, ct, _cal) in pr.stats.items():
        cc_, nc_, tt_, ct_ = by_name.get(name, (0, 0, 0.0, 0.0))
        by_name[name] = (cc_ + cc, nc_ + nc, tt_ + tt, ct_ + ct)

    def g(name):
        return by_name.get(name, (0, 0, 0.0, 0.0))

    def row(label, names, note=""):
        nc = sum(g(n)[1] for n in names)
        ct = sum(g(n)[3] for n in names)
        print(f"  {label:44s} {ct:8.1f}s {100*ct/total:5.1f}% "
              f"calls {nc:>7d}  {note}", flush=True)
        return ct

    print(f"\n[prof] profile total {total:.1f}s; breakdown by "
          f"CUMULATIVE time (parents include children):", flush=True)
    print("--- pipeline phases ---", flush=True)
    fit_boost = row("fit_boosted (production unit)",
                    ["fit_boosted"])
    row("fit() base pass (adaptive)", ["fit"])
    row("_fit_impl (base adaptive loop)", ["_fit_impl"])
    fp = row("fit_poles (ALL calls: optimizer+callbacks)",
             ["fit_poles"])
    trf = row("_trf_gram (vendored TRF loop)", ["_trf_gram"])
    row("fun() residual callbacks (-> core)", ["fun"])
    row("gram() G/g callbacks (-> jac + gram_and_grad)", ["gram"])
    print("--- inside the optimizer ---", flush=True)
    core = row("_varproj_core (QR + X/Res solves)", ["_varproj_core"])
    jac = row("_varproj_jac (D/Z/[K]/S; K skipped on QR path)",
              ["_varproj_jac"])
    gg = row("gram_and_grad (Hadamard G,g assembly)",
             ["gram_and_grad"])
    eigh = row("np.linalg.eigh (trust-region subproblem)",
               ["eigh"])
    sel = row("_gram_select_step + quadratics",
              ["_gram_select_step", "_gram_eval_quadratic",
               "_gram_build_quadratic_1d"])
    print("--- per-call linear algebra (attribution by wrapper) ---",
          flush=True)
    for nm in ("qr", "lstsq", "solve_triangular", "svd", "nnls",
               "eigvalsh", "eig", "solve", "cholesky"):
        nc, = (g(nm)[1],)
        if nc:
            ct = g(nm)[3]
            print(f"    {nm:18s} {ct:8.1f}s {100*ct/total:5.1f}% "
                  f"calls {nc:>7d}", flush=True)
    print("--- PSD machinery ---", flush=True)
    row("psd_dict_init (+ nnls)", ["psd_dict_init"])
    row("psd_refine (accept/reject candidates)", ["psd_refine"])
    row("psd_solve_admm (convex PSD solve)", ["psd_solve_admm"])
    row("project_psd (batched clip)", ["project_psd"])
    row("refit_scales (nnls scale refit)", ["refit_scales"])
    print("--- adaptive-loop machinery (non-optimizer) ---", flush=True)
    row("solve_coefficients (all final/intermediate solves)",
        ["solve_coefficients"])
    row("prune", ["prune"])
    row("suggest_pole_insertions", ["suggest_pole_insertions"])
    row("merge_duplicates", ["merge_duplicates"])
    row("lorentzian_basis (all calls)", ["lorentzian_basis"])
    row("basis_apply (model assembly)", ["basis_apply"])
    row("_quad_rms (candidate rms)", ["_quad_rms"])
    row("compress_channels (once per _fit_impl)",
        ["compress_channels"])
    row("prepare (scale/hermitize)", ["prepare"])

    print("--- derived shares ---", flush=True)
    inner = core + jac + gg
    print(f"  varproj factor work (core+jac+gram_and_grad): "
          f"{inner:.1f}s = {100*inner/total:.1f}% of total, "
          f"{100*inner/fp:.1f}% of fit_poles", flush=True)
    opt = eigh + sel
    print(f"  trust-region subproblem (eigh+select_step): "
          f"{opt:.1f}s = {100*opt/total:.1f}% of total", flush=True)
    print(f"  _trf_gram loop overhead (excl. callbacks/eigh/select): "
          f"{trf - inner - opt:.1f}s "
          f"({100*(trf - inner - opt)/total:.1f}%)", flush=True)

    print("\n[prof] top-25 by TOTTIME (self time):", flush=True)
    ps = pstats.Stats(pr)
    ps.sort_stats("tottime")
    ps.print_stats(25)


if __name__ == "__main__":
    main()
