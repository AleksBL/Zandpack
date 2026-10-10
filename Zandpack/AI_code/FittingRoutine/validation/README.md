# validation/ — test, validation, and benchmark scripts

Collected from the /tmp scratch area of the 2026-10-08/09 sessions.
These scripts are all runnable from anywhere (they insert the repo
root onto sys.path themselves) EXCEPT where noted; the standing rules
apply to all of them:

- `OMP_NUM_THREADS=1` (threaded BLAS is slower at these sizes and
  breaks reproducibility).
- TD_Transport/Device pickles must be loaded BEFORE any repo-root
  `sys.path` insert (Zandpack/siesta_python at the root are
  inspection copies that shadow the installed packages).
- Long-running ones were executed via background processes with
  `... | tee log` so results survive CLI interrupts.

`tr_solver` note (2026-10-09): scripts written before the lsmr
default compare against exact-mode references; pass
`tr_solver='exact'` explicitly when a bit-for-bit comparison against
pre-2026-10-09 stored results is intended.

## 01_shim_and_merge — the AI_fit.py single-file merge verification

| script | what it does |
|---|---|
| check_shim.py | the refine_sampling.py shim re-exports every AI_fit object; package-era fit caches unpickle through it |
| ast_scan.py | AST scan for used-but-never-defined names (found the Gvals/Ginds/GBlock_shape merge bugs) |
| merge_diff.py | token-level diff of every function vs the checkpoint original (found only the intentional _fit->fit) |
| merge_diff2.py | prints the two functions that differed (fit_boosted/_fit_one) for eyeball verification |

## 02_gate_a_bitwise — bit-for-bit gates (old module vs new)

The gate template: import the current AI_fit plus one backup module
from backups/, run identical adaptive fits, require
`np.array_equal` on e/w/C. One script per code-change round, each
pointing at the backup made just before that change:

| script | gates the change |
|---|---|
| gateA_identity.py | the original merge (vs pre_channelcomp) |
| gateA_varproj.py | the _varproj_factors in-place/allocation cleanup |
| gateA_kernel.py | the numba dense-Jacobian kernel |
| gateA_a1.py | the A1 lazy core/jacobian split |
| gateA_step3.py | the step-3 cleanups (theta.copy, one-pass D) |

TRAP embedded in all of them: backup modules with numba
`cache=True` kernels cannot load under dynamic module names — the
harnesses force `old._njit = None` (the numpy J path, bitwise
identical), and stale `backups/__pycache__/*.nbi/*.nbc` indexes must
be deleted after any such crash.

## 03_performance — profiling and speed benchmarks

| script | what it does |
|---|---|
| profile_fit.py | cProfile + runtime-wrapped phase table of the RSSE production fit (located the merge bug; found scipy SVD 66.5%) |
| profile_AI_fit.py | numerical identity gate of the merge (p=64 PSD profile case) |
| step1_profile.py | speedupguide_new.md step 1: time split into core/jac/SVD/other |
| time_ag3au3.py | one-off timing of the ag3au3 benchmark fit |
| trf_vs_dogbox.py | least_squares method comparison on synthetic mixtures |
| op_vs_dense.py | matrix-free VarProjJacobian+lsmr vs dense-jac per solve (found 1.7-2.2x and the quality risks) |
| jac_kernel_bench.py | numpy vs repeat-prealloc vs numba-kernel J assembly (adopted the kernel) |
| validate_varproj.py | bitwise identity + per-call timing of the _varproj_factors rewrite |
| check_integrated_kernel.py | integrated numba kernel vs pre-edit backup, bitwise |
| check_compression_active.py | proved channel compression already active in production (36->12 on RSSE) |
| speed_old_vs_new.py | end-to-end old-vs-new module timing on the RSSE fit (adapt to compare any two backups) |
| speed_jac.py / speed_a1.py | same, for the kernel and A1 rounds |
| speed_jac_nogap.py | the shelved-then-unshelved nogap lead0 A/B (also reproduces fit_boosted from a backup module) |
| nogap_mem.py | nogap fit peak-RSS + sampled RSS curve, current vs backup (proved the memory growth pre-existing, A1 reduced it) |
| nogap_lsmr.py | nogap lead0 under lsmr vs the exact reference |

## 04_guides — varproj_factors_speedup_guide.md evaluation

| script | what it does |
|---|---|
| guide_measure.py | measured 52.3% cost-only _varproj calls; SVD-fallback frequency |
| guide_a1_test.py | A1 lazy split + A3: bitwise identity incl. fallback path, realistic-pattern benchmark (ADOPTED) |
| guide_a6_test.py | A6 unpadded ridge products: fails the exact gate, 0.91-1.07x (REJECTED) |

## 05_lsmr — the tr_solver decision (now the default)

| script | what it does |
|---|---|
| lsmr_smoke.py | per-solve exact vs lsmr at M=24/50/100 (crossover ~M40) |
| lsmr_full_gate.py | 3 alternating A/B full RSSE fits: lsmr 1.55x faster, Gamma rms 11x better |
| lsmr_t_gate.py | T-level relaxed gate: the [3,5) regression and notch improvement |
| verify_bias_vs_ringing.py | savgol bias/ripple split proving the [3,5) lsmr error is a same-sign BIAS, not ringing (lsmr_vs_exact_edge_band.md) |
| validate_lsmr_default.py | post-default-change validation: tests, operator branch, default path |

## 06_nogap_adoption — the nogap_Ag5 system round

| script | what it does |
|---|---|
| inspect_nogap.py / inspect_nogap2.py | pickle/contour/Gamma structure survey (van Hove spike at 3.30, cliff at -3) |
| nogap_diag.py | per-band Gamma/T bias-vs-oscillation diagnosis; exonerated find_correction |
| nogap_eta.py | proved the comparison-eta artifact (T rms 0.355 at eta 1e-2 vs 0.031 matched) |
| seed_sweep.py / seed_stage2.py | the parameter sweep harnesses (w_max 0.3 adopted) |
| adopt_w03.py | wrote the winning sweep fits into cmp_fits.npz |
| compare_summaries.py | field-by-field agreement of regenerated method_summary.json vs stored (all 7 systems) |

## 07_zgnr_and_sampling — zgnr refinement + non-uniform grids

| script | what it does |
|---|---|
| test_sampling_zgnr.py | offline adaptive-sampling validation on zgnr lead0 (truth = 251 pts) |
| zgnr_full_ref.py | full-grid reference fit for comparison |
| zgnr_mineig_analysis.py / zgnr_psd_check.py / zgnr_offwindow_check.py | PSD and off-window decay checks of the refined fits |
| explore_zgnr.py / pair_check.py | pole-pair conjugate-symmetry verification (1e-16) |
| diag_nonuniform.py / diag_skrf.py / nonuniform_test.py / nonuniform_moderate_test.py | non-uniform-grid stress tests (30:1 spacing) |

## 08_psd_and_model — PSD path and model-form checks (earlier rounds)

| script | what it does |
|---|---|
| psd_adaptive_test.py / test_admm_or.py | PSD-projected candidate tracking and over-relaxed ADMM |
| review_synthetic.py | the lorentz_fit_review.md synthetic benchmark |
| check_asym.py / check_equiv.py / test_deriv.py | pole-pair symmetry, pack/unpack equivalence, basis derivatives |
| smoke_boost.py / smoke_verbose.py | residual-boost and verbose-logging smoke tests |
| run_checks.py / run_tests.py / layout_check.py / b_spectrum.py | small harnesses from the review round |

## 09_build_tool

| script | what it does |
|---|---|
| build_AI_fit.py | the merge harness that produced AI_fit.py from the package (reusable if sources change; see its header for the prefix-strip rules and the suggest_pole_insertions rename) |

NOT collected: one-off file patchers (patch*.py) and the abandoned
a6_probe.py draft.

Companion folders already in the repo: tests/ (the ported AI_fit
test suites), Data2/VerySmallRSSE/profiling/ (the RSSE profiling
harnesses + .prof dumps), and the per-system figure scripts
(make_edge_ringing_figure.py, make_lsmr_tradeoff_figure.py) which
live next to their data.
