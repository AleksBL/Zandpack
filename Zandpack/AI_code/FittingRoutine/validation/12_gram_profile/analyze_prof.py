"""Analyze the dumped cProfile stats of the production gram fit
(ag3au3_gram_c0.prof: fit_boosted, Ag3Au3 lead1, gram@1e-9, control
start, 110 s under profiler vs ~100-116 s unprofiled).

Breakdown by cumulative time (parents include children), plus the
top functions by self time.
"""
import os
import pstats

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ps = pstats.Stats(os.path.join(HERE, "ag3au3_gram_c0.prof"))
    total = ps.total_tt
    by_name = {}
    for (fn, _ln, name), (cc, nc, tt, ct, _cal) in ps.stats.items():
        agg = by_name.get(name, (0, 0, 0.0, 0.0))
        by_name[name] = (agg[0] + cc, agg[1] + nc, agg[2] + tt,
                         agg[3] + ct)

    def g(name):
        return by_name.get(name, (0, 0, 0.0, 0.0))

    def row(label, names, note=""):
        nc = sum(g(n)[1] for n in names)
        ct = sum(g(n)[3] for n in names)
        print(f"  {label:46s} {ct:8.1f}s {100*ct/total:5.1f}% "
              f"calls {nc:>7d}  {note}", flush=True)
        return ct

    print(f"profile total: {total:.1f}s "
          f"({len(ps.stats)} distinct functions)", flush=True)
    print("\n--- pipeline phases ---", flush=True)
    row("fit_boosted (production unit)", ["fit_boosted"])
    row("fit() both passes (base + boost residual)", ["fit"])
    row("_fit_impl (adaptive loops)", ["_fit_impl"])
    fp = row("fit_poles (ALL calls)", ["fit_poles"])
    trf = row("_trf_gram (vendored TRF loop, incl. callbacks)",
              ["_trf_gram"])
    row("fun() residual callbacks", ["fun"])
    row("gram() G/g callbacks", ["gram"])
    print("--- inside the optimizer ---", flush=True)
    core = row("_varproj_core (QR + X/Res)", ["_varproj_core"])
    jac = row("_varproj_jac (D/Z/S)", ["_varproj_jac"])
    derv = row("basis_derivatives (dPhi per theta)", ["basis_derivatives"])
    gg = row("gram_and_grad (Hadamard G,g)", ["gram_and_grad"])
    eigh = row("np.linalg.eigh (trust-region subproblem)", ["eigh"])
    sel = row("_gram_select_step + quadratics",
              ["_gram_select_step", "_gram_eval_quadratic",
               "_gram_build_quadratic_1d"])
    print("--- per-wrapper linear algebra ---", flush=True)
    for nm in ("qr", "lstsq", "solve_triangular", "svd", "nnls",
               "eigvalsh", "eigh", "solve", "cholesky", "matmul",
               "einsum", "tensordot", "dot"):
        if g(nm)[1]:
            cc, nc, tt, ct = g(nm)
            print(f"    {nm:18s} {ct:8.1f}s {100*ct/total:5.1f}% "
                  f"calls {nc:>7d}  (self {tt:.1f}s)", flush=True)
    print("--- PSD machinery ---", flush=True)
    row("psd_dict_init (+nnls)", ["psd_dict_init"])
    psdr = row("psd_refine", ["psd_refine"])
    admm = row("psd_solve_admm", ["psd_solve_admm"])
    row("project_psd", ["project_psd"])
    row("refit_scales", ["refit_scales"])
    print("--- adaptive-loop machinery ---", flush=True)
    sc = row("solve_coefficients", ["solve_coefficients"])
    row("prune", ["prune"])
    row("suggest_pole_insertions", ["suggest_pole_insertions"])
    row("merge_duplicates", ["merge_duplicates"])
    row("lorentzian_basis", ["lorentzian_basis"])
    row("basis_apply", ["basis_apply"])
    row("_quad_rms", ["_quad_rms"])
    row("residual_and_jacobian (if any)", ["residual_and_jacobian"])
    row("_assemble_dense_j (if any)", ["_assemble_dense_j"])

    print("\n--- derived shares ---", flush=True)
    inner = core + jac + gg + derv
    print(f"  varproj factor work (core+jac+d/dtheta+gram_and_grad): "
          f"{inner:.1f}s = {100*inner/total:.1f}% of total, "
          f"{100*inner/max(fp, 1e-9):.1f}% of fit_poles", flush=True)
    opt = eigh + sel
    print(f"  trust-region subproblem (eigh+select_step): {opt:.1f}s "
          f"= {100*opt/total:.1f}%", flush=True)
    print(f"  TRF loop overhead (excl. callbacks/eigh/select): "
          f"{trf - inner - opt:.1f}s "
          f"({100*(trf - inner - opt)/total:.1f}%)", flush=True)
    print(f"  fit_poles share of the whole fit_boosted: "
          f"{100*fp/total:.1f}%", flush=True)
    psd_all = psdr + admm
    print(f"  PSD machinery (refine+admm): {psd_all:.1f}s "
          f"= {100*psd_all/total:.1f}%", flush=True)

    print("\n--- top-30 by SELF time (tottime) ---", flush=True)
    ps.sort_stats("tottime")
    ps.print_stats(30)


if __name__ == "__main__":
    main()
