# Handoff — lorentz_fit: shared-pole Lorentzian fit of Hermitian matrix functions

Status: working, benchmarked, and applied with transport-level
(T/DOS) validation to every Data2 system WITH pickles (section 6).

RENAME (2026-10-10, user-directed): the active implementation is
`AI_fit_base.py` (was AI_fit.py), the sweep framework is
`AI_fit_sweep.py` (was sweep_framework.py), and the sampling/glue
API surface is `AI_fit_adaptivesampling.py` (was refine_sampling.py).
The OLD names survive as MODULE-ALIAS shims (sys.modules aliasing —
`import AI_fit`, `import refine_sampling` keep working, pickled
caches keep unpickling); remove them only after regenerating every
cache. Historical sections below may still use the old names —
the aliases make them equivalent. Verification after the rename:
both test suites ALL PASS, both cache generations unpickle,
alias identity checked.

The ACTIVE implementation is the single file `AI_fit_base.py` (the
`lorentz_fit/` package is legacy, kept for old fit-cache
unpickling). The 2026-10-09/10 rounds replaced the per-system
hand-tuned workflow with a MEASURED, repeatable one:

- `AI_fit_sweep.py` (repo root; was sweep_framework.py): health-gated multi-start
  parameter sweeps + T/DOS evaluation (4.15; usage GUIDE section 9).
- `GUIDE_3py_integration.md` (repo root): the complete 3.py porting
  guide (template, refinement recipes, pole inheritance for
  symmetric systems, framework section).
- Best matched-eta T rms per system (all history in 6.3): Ag3Au3
  0.0062, Ag3Au3_gap 0.0035, Fig2Ag5 0.0053, spinpol_ag5 0.0079
  (lowpole 50-pole budget), nogap_Ag5 0.0298, agnr 0.0544, zgnr
  0.089, zgnr_spin 0.131, VerySmallRSSE 0.020. THREE stored suite
  comparisons (spinpol, Fig2Ag5, zgnr_spin) were inflated ~5x by an
  eta mismatch (teta != contour eta) - always compare matched
  (4.15/4.16). The Ag3Au3-family 'exact' pin is RESOLVED: gram +
  multi-start beats or ties it at ~5x less wall (section 8).
- Fits are ~4-9x faster after the 4.11 performance round and another
  ~3x from the gram solver + refit_scales/PSD-fusion rounds; the
  4.13 spread round established that single-start comparisons carry
  no information on basin-sensitive systems (joint health gate:
  padded min eig AND T-level error together). gram_cut=1e-9
  confirmed bit-identical to cut=0.

This document is the SINGLE handoff (project_handoff.md is retired
— its content lives in 4.11/4.13/4.14). The previous project
iteration is preserved in `attempt-1/` (see `attempt-1/HANDOFF.md`
for its results and dead ends).

---

## 1. What is fitted

Given samples F(E_n) ∈ C^{p×p}, F(E_n) = F(E_n)*, n = 1..N (the lead
level-width matrices Γ of the Zandpack pipeline), the code returns the
model of the implementation plan and nothing else:

    F_fit(E) = Σ_{i=1}^{M} C_i · w_i² / ((E − e_i)² + w_i²),    C_i = C_i*

- shared pole pairs (e_i, w_i) across ALL matrix entries,
- Hermitian residue matrices C_i (exact by construction),
- strictly proper decay: no constant, linear, or dispersive terms
  (verified by `tests/test_model_form.py` to machine precision).

The model form is guaranteed at the API level: `FitResult.evaluate(E)` is
exactly the formula above (test compares it against an independent
implementation from the raw outputs: 3e-16 relative deviation).

## 2. Directory layout

```
FittingRoutine/
  AGENTS.md                     project instructions
  implementation-plan-...md     the design document this implements
  HANDOFF.md                    THIS file
  GUIDE_3py_integration.md      the self-contained user guide for
                                porting a Data2 system folder's 3.py to
                                the new pipeline (template incl. the
                                multi-start health-gate helpers, solver
                                table, verification, pitfalls; 4.10's
                                expanded companion)
  AI_fit_base.py                the WHOLE routine as ONE module (the
                                migration-ready single file: all of
                                lorentz_fit + refine_sampling; tests
                                tests/AI_fit_test_*.py pass against it
                                and it reproduces the package numbers)
  migration_plan_zandpack.md     CURRENT migration plan (2026-10-10
                                rewrite, NESTED-PACKAGE setup): target
                                Zandpack/AI_code/FittingRoutine/ (a
                                package; alias registration in
                                __init__.py keeps old pickles working);
                                delivers AI_fit_base.py + AI_fit_sweep.py
                                + AI_fit_adaptivesampling.py + the
                                docs/ .md set + the validation gate
                                scripts (artifacts excluded) + the
                                distilled 4.15/4.16 content into the
                                docstring. The package-era original is
                                preserved at
                                backups/migration_plan_zandpack_2026-10-
                                08_package_era.md
  project_handoff.md            RETIRED stub - the gram-solver round's
                                focused handoff, merged into sections
                                4.11/4.13/4.14 (2026-10-09)
  performance_guides_outcome.md what worked / what did not, from the
                                archived guides (below)
  .junk/                        archived planning documents:
                                lorentz_fit_review.md (drove 4.9),
                                lorentz_fit_performance_guide{,_2}.md
                                (drove 4.11), secondaryplan.md
  lorentz_fit/                  the package (LEGACY — kept only so that
                                package-era fit caches unpickle; no
                                active code imports it. Final package
                                state archived in
                                checkpoint_2026-10-08_lorentzfit/)
    sampling.py                 adaptive grid refinement proposals (numpy-only;
                                fit-based, leave-one-out removal, and the
                                fit-free derivative mode)
  AI_fit_base.py                THE ACTIVE IMPLEMENTATION (2026-10-08
                                evening): the single-file merge of the
                                lorentz_fit package + the historical
                                refine_sampling.py (see the RENAME note in
                                the header).
                                Everything (fit, sampling proposals,
                                extract_dense, inject_fits, resample,
                                refine_loop, sampled_transmission_dos,
                                fit_boosted, fit_fixed_poles) lives
                                here. New code imports this directly.
  AI_fit_sweep.py               REUSABLE SWEEP TOOL (2026-10-09, was
                                sweep_framework.py):
                                GammaSweep (multi-start, health-gated,
                                cached fit sweeps) + evaluate_pair
                                (T/DOS vs reference + plots). Usage
                                guide: GUIDE_3py_integration.md 9;
                                provenance: the nogap_Ag5 rounds.
  AI_fit_adaptivesampling.py    MODULE-ALIAS of AI_fit_base (2026-10-10
                                rename; was the refine_sampling.py shim):
                                the sampling/glue API surface.
  AI_fit.py / refine_sampling.py COMPAT ALIASES (sys.modules aliasing of
                                AI_fit_base / AI_fit_adaptivesampling):
                                keep old imports and pickled caches working.
                                Remove only after regenerating every cache.
  tests/
    test_synthetic.py           optimizer/Jacobian/PSD/off-window tests
    test_model_form.py          proof that the model is plan-exact
  bench/
    extract_data.py             pickle -> dense (E, F) loaders
    run_fit_bench.py            metric-collecting benchmark (JSON out)
    plot_examples.py            sampled-vs-fit figures (per example)
    oscillation_check.py        full-suite fit-vs-PCHIP ripple check
    performance_figure.py       aggregate performance overview figure
    results_lorentz_fit/        benchmark JSONs + figures/
  Data2/                        second-generation examples WITH the sampling
                                scripts (1.py/2.py/3.py) and INSTRUCTIONS.md
    apply_method.py             generic driver: applies the current method to a
                                system and compares against its own old 3.py
                                (legacy single-start path; the sweep rounds
                                4.15 supersede its stored teta-mismatched
                                comparisons for 3 systems - see 6.3 caveat)
    README_remaining_round.md   the 2026-10-10 five-system sweep-round record
                                (zgnr, zgnr_spin, Ag3Au3, Ag3Au3_gap, RSSE)
    <system>/sweep_*/           per-system sweep caches (compressed npz),
                                results jsons, gamma rankings, fig_*.png
                                plots; README_sweep.md / README_gram_refined.md
                                per system carry the full round records
    zgnr/, zgnr_spin/, VerySmallRSSE/
                                the deep-dive systems (refinement drivers,
                                dos_t_comparison.png, extend_window.py ...)
    agnr/, spinpol_ag5/, Fig2Ag5/, Ag3Au3/, Ag3Au3_gap/, AuBreakJunction/
                                one-shot applications via apply_method.py
                                (method_comparison.png + method_summary.json)
  attempt-1/                    previous iteration (untouched reference; its
                                lorentzmesh.py is used only to replicate the
                                old Ag3Au3 3.py pipeline verbatim)
  Data/                         example pickles (+ AuBreakJunction has none)
  Block_matrices/, Zandpack/,  reference code excerpts (INSPECTION ONLY —
  siesta_python/                never import; the installed packages are the
                                real code. See section 3 shadow warning.)
```

## 3. Environment and dependency tree

Interpreter: `/home/aleks/oak/bin/python3` (Python 3.14).

```
lorentz_fit (runtime)
  ├── numpy 2.5.2        (linalg: lstsq, qr, eigh, einsum)
  ├── scipy 1.18.1
  │     ├── optimize.least_squares   (trf, bounded, outer pole fit)
  │     ├── optimize.nnls            (PSD scale refit)
  │     ├── signal.find_peaks        (residual-driven pole insertion)
  │     └── sparse.linalg.LinearOperator  (rank-1 Jacobian, see 4.3)
  └── scikit-rf 2.1.0    (INITIALIZATION ONLY: vector_fit pole relocation)
        note: import path `from skrf.vectorFitting import VectorFitting`
        (skrf.VectorFitting does NOT exist at package top level)

bench (plotting only)
  └── matplotlib 3.11.1  (Agg backend)

NOT used: cvxpy (not installed; PSD path uses projection + nnls instead),
numba, scikit-rf residues (only its relocated poles are adopted).

External inputs
  ├── Data/<example>/tmp.gz (gzip) or C1.xz (lzma): pickled TD_Transport
  │     instances. Loaded with plain pickle; extraction needs
  │     R.Contour, R.sampling_idx, R.Nonortho_Gammas (block-sparse:
  │     .inds = block index pairs, .vals[block][ik][sample_idx]).
  └── attempt-1/bench_cache/AuBJ_lead{0,1}.npz: AuBreakJunction has NO
        pickle; attempt-1 pre-extracted its Gamma blocks (p = 85/90!).

CRITICAL environment trap: the local excerpt folders `Zandpack/` and
`Block_matrices/` SHADOW the installed packages as soon as the
FittingRoutine folder is on sys.path (even appended, even via cwd for
`python3 -`). Unpickling then fails. Consequence: ALWAYS load the
pickles BEFORE any sys.path manipulation and run bench scripts from
`bench/` (all provided scripts do this correctly).
```

## 4. How the code works

### 4.1 Data preparation (`data.py`)

`prepare(E, F)` verifies shapes/finiteness, Hermitizes (F + F^H)/2 and
measures the violation, then rescales: energies to [-1, 1]
(E' = (E−mid)/scale) and amplitudes to O(1) (f_scale = max_n ||F_n||_F).
All fitting happens in scaled units; `FitData` carries the conversion and
`FitResult` returns raw-unit e, w, C (scaled copies kept in the result).

### 4.2 Pole initialization (`poles_init.py`) — plan option A

skrf's VectorFitting pole relocation runs on the stacked matrix entries;
its fitted conjugate pole pairs become the starting (e_i, w_i); residues
are discarded. Domain mapping (verified against the skrf 2.1.0 source):

- skrf fits G(s) = Σ [r/(s−p) + r̄/(s−p̄)], s = j·2π·f, one stored pole
  per conjugate pair (Im ≥ 0, Re flipped ≤ 0), normalized by mean(f).
- We feed f = E + σ (σ = half-span − Emin, scaled) as positive Hz and
  the Hermitian samples as S-parameters.
- Extraction: e = Im(p)/2π − σ, w = −Re(p)/2π; the conjugate partners map
  to mirrored far-off centers (−e−2σ) that only carry smooth background
  and are dropped. Never feed negative frequencies (conjugate-pair
  symmetry would force F to be even).
- `vf.max_iterations = 10` (default 100 dominates runtime at large p);
  `fit_constant=False, fit_proportional=False` (strict decay).
- For p > max_ports (32): relocation runs on the principal submatrix of
  the strongest diagonal entries — poles are shared across entries, so
  this is a valid initializer at a fraction of the cost.
- Fallback: `linspace_init` (equidistant centers, 2% widths), used when
  skrf is missing or degenerates (reported on stderr, never silent).

### 4.3 Variable projection (`varproj.py`) — the core

At fixed (e_i, w_i) the model is entrywise linear; each Hermitian
upper-triangle entry is a real channel (p diagonal + 2·p(p−1)/2 re/im
channels — the lower triangle is never fitted redundantly). All channels
are solved in ONE `lstsq` call (shared SVD/QR).

- **Exact analytic Jacobian** (Golub–Pereyra, including the term
  Kaufman's simplification drops):
  dR/dθ_j = (I−P) dA_j β − A(AᵀA)⁻¹ dA_jᵀ R.
  QR fast path (X = solve(R1, QᵀY), K = solve(R1, Qᵀ).T — note: NOT
  solve(R1.T, ...)); falls back to rank-truncated SVD when QR is
  ill-conditioned (diag(R1) ratio < 1e-12 — near-coincident poles).
  Verified against central finite differences to 1.6e-9.
- **LinearOperator Jacobian** (`VarProjJacobian`): each Jacobian column
  is a difference of two rank-1 terms, so matvec/rmatvec are ~4 ms
  instead of materializing a (N·n_ch × 2M) dense J (up to 336 MB at
  p = 27). Used when N·n_ch·2M > 3e6. least_squares then uses trf+lsmr;
  x_scale is a fixed array of exact column norms at θ₀ ('jac' scaling is
  forbidden with operators). Operator verified against dense J to 7e-16.
- **Tikhonov ridge** on coefficients, implemented EXACTLY by augmenting
  the design with [λI; 0] rows (the GP derivative machinery stays exact
  for the augmented problem). λ = 0.05 default. Two-phase: ridge = 0
  during the adaptive optimization (the regularized landscape makes trf
  wander: measured 400 nfev without convergence, 5× slower), ridge ON in
  the final polish and final coefficient solve. It only bites near-null
  directions of the sample LS (AᵀA ~ λ²); legitimate directions are
  affected at ~1e-5.

### 4.4 Adaptive loop (`__init__.py::_fit_impl`, `prune.py`)

1. fit at the skrf-initialized poles (trf, bounded, ≤400 nfev per call),
2. prune negligible residues (rel_tol 1e-3) and refit,
3. insert poles at interior residual peaks (scipy find_peaks with
   prominence, sub-sample parabolic peak refinement, width = local
   spacing; insertions must be ≥ 1 sample spacing from existing centers),
4. repeat until target_rms or max_poles (bench cap: 60 pairs),
5. final polish (≤1500 nfev) at the best poles,
6. duplicate cleanup (below), final weighted coefficient solve.

### 4.5 The anti-spike defense stack

Root cause of all observed artifacts (off-window peaks, inter-sample
λ_min dips): near-null directions of the sample LS — coefficient
combinations that vanish ON the sampled energies but not between/beyond
them. Five mechanisms, by spike flavor:

| flavor | mechanism | defense |
|---|---|---|
| narrow pole outside window | peak invisible to samples; tail fixes edge residual | centers within ±15% of window (e_margin) |
| unconstrained tails beyond window | no data off-window | tail anchors |
| sub-resolution pole pair (w ≈ 0.075·ΔE) | sample columns nearly parallel; residues bounded only by tiny tails | width floor (two-stage, below) |
| window-spanning broad cluster | near-degenerate smooth basis | ridge in final solve |
| exactly coincident cancelling pair (‖C‖~10¹⁰) | exact null direction | QR conditioning check + merge |

- **Tail anchors** ("artificial ramps"): 8 zero-target pseudo-samples at
  ±[0.05, 0.15, 0.35, 0.65]·(span/2) beyond the edges, weight
  0.05 × median sample weight, appended AFTER the real samples. Pole
  insertion only ever sees the real-sample residual slice.
- **Two-stage width floor** (public `fit()` dispatches `_fit_impl`):
  tight floor = max(1e-3, 0.5·median spacing) first — stable for smooth
  data (smallrsse); if rms > 3·target AND > 3e-3 the data has sharp
  resolvable structure → redo with the loose absolute floor 1e-3
  (fig2ag5, spinpol). `result.history["w_min"]` records the choice.
- **Local width floor (non-uniform grids)** — stage 4b of `_fit_impl`,
  `local_floor_frac=1.0`: after merging, any pole whose width is below
  `local_floor_frac × (distance from its center to the nearest sample)`
  is widened to that local floor and briefly refit (accept only if rms
  stays within the merge guard). On a uniform grid the local floor is
  half the spacing (identical to the tight floor), but on refined
  grids the median spacing no longer describes the sparse regions and
  the median-based floors let spike poles hide between samples — the
  stress test (30:1 spacing-ratio grid) measured a dense-grid error of
  1.6× the function scale with sample rms looking perfect; with the
  local floor it is 0.15×. Realistic refined grids (5–10× local
  density at peaks) sit at 0.075×. `result.history
  ["n_widened_subresolution"]` counts widened poles; 0.0 disables.
- **merge_duplicates**: merges pairs that coincide to <5% of width AND
  whose residues cancel to <15% (true null artifacts; 50% ate
  legitimate close poles and cost 30× rms). Residue-norm-weighted merge;
  a merge is only accepted if rms stays within 2× (a buggy earlier
  version accepted degraded merges — fixed and guarded).

### 4.6 Constraints and validation

- Hermiticity: exact by construction (`tests` assert it).
- PSD: `fit(psd=True)` runs `constraints.psd_refine` — per-pole
  eigenvalue clip to ≥0 plus a non-negative shared-scale refit (nnls).
  Because unconstrained fits rely on inter-pole cancellation (attempt-1
  measured the same), this can cost orders of magnitude in rms; the
  refined model is KEPT only if it does not explode the error, and
  `result.psd["applied"]` says which model you got. NOTE: what matters
  physically is λ_min of the assembled total fit; the fit's total λ_min
  is near-PSD in-window on well-resolved data even without the
  projection (see benchmark figures).
- `validate.py`: held-out error (coefficients retrained on train split),
  λ_min of the total fit on dense grids, complex-plane limsup
  (hidden-cancellation detector), dispersive-basis diagnostic (tests
  whether the absorptive-only basis is adequate for the data).

### 4.7 API reference

```python
import AI_fit as lf                  # ACTIVE single-file module
res = lf.fit(E, F,                   # (N,), (N,p,p) Hermitian samples
          n_poles=None,               # exact count; None = adaptive
          n_init=32,                  # initial pole pairs (overinit)
          target_rms=5e-3,            # adaptive stopping target
          max_poles=None,             # cap (default 4*n_init)
          weight="relative",          # per-sample 1/||F_n||
          init="skrf",                # or "linspace" (NO-OP under
                                     # psd=True: the dispatch always
                                     # uses the NNLS PSD dictionary)
          init_poles=None,            # optional (e, w) seed set; bypasses
                                     # the automatic init - for basins
                                     # the relocators cannot reach
          psd=False,                  # PSD refinement (see 4.6)
          psd_force=False,            # accept the PSD model even at
                                     # rms cost (physical Gamma >= 0
                                     # everywhere; the workflow default)
          channel_tol=0.0,            # drop channels < tol*max (large p)
          e_margin=0.15,              # center range beyond window
          tail_anchors=True, anchor_weight=0.05,
          ridge=0.05,                 # final-solve Tikhonov
          w_min=None,                 # None = two-stage floor
          local_floor_frac=1.0,       # non-uniform-grid width floor
          eta_raw=None,               # imaginary broadening of the data;
                                     # poles narrower than mult*eta are
                                     # forbidden (see 4.9)
          eta_floor_mult=1.0,         # floor = mult * eta (see 4.9; the
                                     # right value is SYSTEM-DEPENDENT -
                                     # never above 1 on the tested
                                     # systems, 1.5 is a dead zone)
          w_max_raw=None,             # width CEILING (0.25-0.5 eV in the
                                     # workflow; sharp optimum, and it
                                     # SHIFTS with the pole budget)
          tr_solver="lsmr",           # or "exact" or "gram" (4.14; gram
                                     # = fastest, opt-in)
          gram_cut=1e-9,              # gram weak-direction truncation
                                     # (1e-9 bit-identical to 0)
          verbose=False)              # full pipeline narration
res.e, res.w, res.C                   # raw-unit poles and residues
res.evaluate(E_grid)                  # model, raw units, (n,p,p)
res.rms, res.n_poles, res.history, res.psd
res.data                             # FitData (scaling, scaled arrays)
res.e_scaled, res.w_scaled, res.C_scaled, res.chans

# FIXED-POLES coefficient solve (pole inheritance for symmetric
# systems, ridge=0.0 default - see 4.15 fixed-poles finding):
res1 = lf.fit_fixed_poles(E, F, f0.e, f0.w, psd=True, psd_force=True)
```

Module map (all inside the single `AI_fit_base.py`, section banners
mirror the legacy package): SECTION data (prepare/FitData), varproj
(basis, channels, packing, linear solves, exact Jacobian +
operator, fit_poles, compress_channels + basis_apply - 4.11),
poles_init (skrf/linspace init + the NNLS PSD dictionary init),
prune (prune, insertion, merge_duplicates, local-spacing helpers),
constraints (PSD: project_psd, over-relaxed ADMM, psd_refine,
fit_fixed_poles), validate (checks), sampling (grid refinement
proposals, incl. the fit-free derivative mode), pipeline (fit
dispatcher + _fit_impl + fit_fixed_poles + FitResult/CombinedFit/
PaddedFit), TD glue (extract_dense, inject_fits, resample,
refine_loop, sampled_transmission_dos, fit_boosted,
equalize_pole_counts). `AI_fit_adaptivesampling.py` at the repo
root is a MODULE-ALIAS of AI_fit_base (the sampling/glue surface);
the compat aliases `AI_fit.py` / `refine_sampling.py` keep old
imports and pickled fit caches working (old caches reference the
module names 'AI_fit' / 'refine_sampling' — import either alias
before unpickling them).

Driver-level API:

```python
from AI_fit import (extract_dense, inject_fits, resample,
                    refine_loop, sampled_transmission_dos,
                    fit_boosted, fit_fixed_poles,
                    equalize_pole_counts)

fits = fit_boosted(E, F, base_kw, n_res=8)   # adaptive PSD fit + a
        # PSD-forced correction fit of the residual (4.9); returns a
        # CombinedFit (concatenated poles, injectable). WARNING:
        # n_res=0 runs an UNCAPPED residual fit - use plain fit() for
        # "no boost" (4.15 lowpole round).
inject_fits(R, fits)          # setLval + FitNO2O (4.7.1); also sets
                              # R.NumL for the downstream helpers
out = refine_loop(R, D, ..., derivative_only=False, deriv_tol=3.0)
# 4.8; derivative_only=True = fit-free insertion rounds from
# d(sum_ij |F_ij|)/dE (steep flanks get density, no fits until the
# grid settles) - the recommended mode (do_remove=False) on
# deliberately dense sharp-feature grids (4.15)
E, T, dos = sampled_transmission_dos(R)   # exact reference from the
#  STORED self-energies (same sub-orbital space; Fermi-pole points
#  excluded via sampling_idx)
```

### 4.7.1 CRITICAL data-assembly note (lower-triangle block storage)

The stored `Nonortho_Gammas` may keep only the LOWER TRIANGLE of the
block structure — e.g. blocks (0,0), (1,0), (1,1) with NO (0,1)
partner. This is true for most examples (all Data/ ones with missing
(1,0); zgnr/VerySmallRSSE-style). Two consequences, both bit us:

- EXTRACTION: a naive Hermitization `F = (F + F^H)/2` HALVES the
  off-diagonal blocks: the missing partner contributes zero, so
  stored-block b10 becomes (b10 + 0)/2. Both `bench/extract_data.py`
  and `refine_sampling.extract_dense` now complete missing partners
  with the conjugate transpose BEFORE symmetrizing (a no-op when the
  block set is already complete, e.g. zgnr).
- INJECTION: the old `R.Fit` containers carry the FULL Hermitian block
  set (it adds the missing partners), and the transport machinery
  (`Blocksparse2Numpy` -> dense Gamma) needs them. `inject_fits`
  creates the container skeleton with the completed block set and
  writes the coefficients through the class's own API:
  `R.setLval(lead, I, J, i, j, val, ik, iL)` per (block, sub-block)
  component — setLval fills the Hermitian partner (J,I) with the
  conjugate itself — followed by `R.FitNO2O()`. The dense p x p fit
  matrices are mapped back through the I,J block / i,j sub-block
  indices (the layout `extract_dense` packed them with).

### 4.7.2 Equal pole counts across k-points (REQUIRED by Zandpack)

The rest of the Zandpack code needs EVERY k-point (and every lead) to
carry the SAME pole count: `R.NumL` is one scalar and the container
helpers (`add_broadening`, `add_move_center`, setLval's default iL)
index every lead with it. A per-(lead, ik) fit naturally produces
different counts (e.g. zgnr_spin: 32/36/38/36).

- `inject_fits(R, fits, equal_nl="all")` — the DEFAULT since
  2026-10-09 — pads every (lead, ik) pole array to the GLOBAL max
  with INERT poles: duplicates of pole 0 with ZERO residues (the
  mechanism inject_fits has always used for per-ik padding; real
  width, NaN-safe, assembled Gamma unchanged). Validated on
  zgnr_spin: all containers (2, 38), R.NumL 38, T curves
  bit-identical to the legacy injection (max diff exactly 0.0).
  `equal_nl="lead"` reproduces the OLD per-lead widths for legacy
  bit-comparisons ONLY — it violates the R.NumL contract when the
  leads' maxima differ (the pre-fix behavior).
- Post-fit processing (e.g. before pickling the fits themselves):
  `equalize_pole_counts(fits, mode="dup", n_target=None)` returns a
  new dict with every fit padded to a common count (default: the
  max; `n_target` overrides). `pad_poles(fit, n, mode)` pads one
  fit. Both modes are purely additive — `evaluate()` is identical
  to the input (validated). mode="dup": duplicate pole 0, zero
  residues (NaN-safe everywhere). mode="zero": zero-WIDTH poles at
  max|e| + 1e3 (basis identically 0; the far center avoids the 0/0
  at E == e — do not evaluate a grid point exactly there).
  The padded objects are `PaddedFit` views: unknown attributes
  delegate to the base fit, so rms/history/psd keep working.
  TRAP: `block_sparse.info()` marks all-val-zero blocks `is_zero=0`
  at CONSTRUCTION, and later `BDot`/`block_MATMUL` skips them — a
  zero-filled skeleton makes `FitNO2O` crash with an empty product.
  The skeleton is therefore initialized with ONES (as Zandpack's own
  `initialize_lorentzian_fit` does) and fully overwritten by setLval.
  `R.NumL` is set to the largest pole count (setLval's default iL and
  the add_broadening helpers use it); per-lead iL is passed
  explicitly. Verified bit-for-bit against the previous direct-array
  injection (identical transmission numbers).

Symptom of getting this wrong (measured on VerySmallRSSE): the fitted
transmission comes out offset by ~+1 over the whole window (T > 1) —
half the off-diagonal lead coupling is missing.

### 4.8 Adaptive sampling refinement (`sampling.py` + `refine_sampling.py`)

The Data2 task: refine the ENERGY SAMPLING GRID of the level-width
function alongside the fit — add points where the sampling
under-resolves the function, drop points that carry no information.
The fit itself is grid-agnostic (see the local width floor above), so
refined (non-uniform) grids are fully supported.

Pure-numpy proposal logic (`lorentz_fit/sampling.py`):

- `suggest_insertions(E, F, tol=5e-3, max_add, poles=(e,w), ...)` —
  flags an interval when the midpoint discrepancy between the
  piecewise-linear and the local 4-point-Lagrange-cubic interpolant
  exceeds tol × max(local ‖F‖, 1e-3 × global scale); splits flagged
  intervals at the midpoint (worst first under a cap). Optional extra
  sources: fit-residual peaks and bracketing of barely-resolved fit
  poles (w < pole_factor × local spacing → two points at e ± w).
- `suggest_removals(E, F, tol=1e-2, max_gap_factor=3)` — leave-one-out
  test: an interior sample is redundant when the 4-point Lagrange
  interpolant through its neighbors reproduces it within tol × local
  scale AND the gap left behind stays below max_gap_factor × median
  spacing. Edges are never dropped.
- `suggest_insertions_derivative(E, F, tol=3.0, max_add)` — FIT-FREE
  insertion proposals from the derivative of the abs-summed function
  s(E) = sum_ij |F_ij(E)|: intervals scored by max|ds/dE| at their
  endpoints, steepest first, gated at tol × median|ds/dE|. Steep
  flanks (band-edge onsets, notch walls) get density without any fit.
- `refine_grid(E, F, ...)` — one-round convenience wrapper.

Zandpack wiring (`refine_sampling.py`, repo root):

- `extract_dense(R, lead, ik)`, `load_pickle(path)`,
  `current_lines(R)` — data access (mirrors bench/extract_data.py).
- `resample(R, D, lines, eta, N_F, read_kwargs, runner=...)` — rebuild
  the contour from per-lead real energies (`R.Make_Contour`), assign
  `D.custom_tbtrans_contour = R.Contour`, rewrite the fdf (`D.fdf()`),
  rerun TBtrans (`D.run_tbtrans_in_dir(overwrite_prev=True)`) and
  `R.read_data(**read_kwargs)`. `TS_TBT.fdf` is snapshotted and
  restored per round (write_more_fdf APPENDS). A custom `runner`
  replaces the whole evaluation step (used by the spin-polarized
  mimic_tbtrans_spin path).
- `refine_loop(R, D, ..., initial_lines=..., n_rounds=3, do_remove=True,
  derivative_only=False, deriv_tol=3.0)`
  — per round: fit every (lead, ik), propose insertions, take the UNION
  of all leads' proposals as the new shared line (TBtrans evaluates all
  leads at every contour point anyway), resample; then one removal
  round (INTERSECTION of the per-(lead, ik) redundant sets — a point
  is only dropped when no lead needs it); final fits for every
  (lead, ik). Returns `{fits, history, removed, lines}`.
  With `derivative_only=True` NO fits run during the insertion
  rounds: the proposals come purely from
  d(sum_ij |F_ij|)/dE (suggest_insertions_derivative). The removal
  stage is fit-free in both modes, and the final fits still run once
  the grid settles — so the mode costs one TBtrans round per round
  and nothing else. Use it as a cheap pre-pass (steep flanks get
  dense first), then a normal fit-based loop, or standalone when the
  fits are too expensive per round (large-channel systems).

CRITICAL import discipline: the repo-root `Zandpack/`,
`siesta_python/`, `Block_matrices/` folders are INSPECTION COPIES and
shadow the installed packages as soon as the repo root is on `sys.path`
(the copy is even missing its `funcs` dependency).
With the single-file `AI_fit.py` this entire import-path bug class is
gone (one self-contained module, no package paths). The
`refine_sampling.py` name is kept as a shim re-exporting AI_fit for
the drivers and pickled caches; `AI_fit.py` itself needs nothing from
`sys.path` beyond the repo root. Example scripts still follow one
rule: load the TD_Transport/Device pickles FIRST (clean `sys.path`),
then import `refine_sampling`/`AI_fit` through a temporary root
insert that is removed immediately after.

Example drivers (Data2): `zgnr/refine.py` (TBtrans path, 3-pole-mode
demonstration: coarse 63-point start → 100 refined points vs the
original 251), `zgnr_spin/sample.py` + `refine.py` (mimic_tbtrans_spin
runner, two spins), `VerySmallRSSE/refined/{make_device.py, refine.py}`
(rebuild-from-scratch per round: the RSSE electrode self-energy is
recomputed by re-running a parameterized make_device.py with the
current energies.npy — the in-place resample does not apply there).

### 4.9 Oscillation fixes (review, Tier 1 + 4.7; archived in .junk/)

The review (archived at `.junk/lorentz_fit_review.md`) identified two distinct
oscillation phenomena: fast ripple in the unconstrained fit (the
objective only sees the samples; the width floor was at the aliasing
limit) and slow undulation in the PSD-forced fit (the PSD step was a
heuristic; poles were optimized for the wrong model). Implemented:

- **Quadrature objective (review 4.1)**: sample weights carry the
  trapezoid width around each sample (`_quad_widths`), and ALL internal
  model-selection metrics use the quadrature rms (`_quad_rms`) - the
  cost approximates an energy integral, so refined (non-uniform) grids
  no longer over-weight dense clusters. Weight semantics (review 4.4):
  kept the half-relative 1/max(||F_n||, eps) deliberately - the fully
  relative 1/||F_n||^2 was tried and measurably degraded peak fidelity
  on data spanning ~4 orders of magnitude.
- **Width floor (review 4.3)**: tight floor raised from 0.5x to 1.0x
  the median spacing (a pole at half the spacing has ~1 sample across
  its FWHM and rings between samples), plus `fit(..., eta_raw=...)`:
  poles narrower than the imaginary broadening of the sampled data are
  forbidden (structure narrower than eta cannot exist in the data).
  The refinement driver passes its contour eta automatically.
  `local_floor_frac=1.5` is used in the comparison workflow.
- **PSD path (review 4.6/4.7)**: (a) the PSD coefficient solve at
  fixed poles is now an exact convex ADMM solve (`psd_solve_admm`,
  warm-started from the clip + scale refit) - the PSD-OPTIMAL residues
  at the given poles; (b) pole initialization for psd fits uses a
  NON-NEGATIVE Lorentzian dictionary NNLS fit of Tr Gamma
  (`psd_dict_init`) - a sparse pole set natively compatible with
  positivity (no cancellation needed); (c) the whole adaptive loop
  tracks candidates by their PSD-projected rms, and a PSD insertion
  pass adds poles at the projected model's residual peaks (two
  variants per round: coefficients-only at the augmented poles - no
  drift back to the unconstrained optimum - and a short local refit).
- **Width policy (user-directed, final)**: `fit(..., w_max_raw=0.5)` -
  no pole wider than ~0.5 eV. A window-span pole (observed: w=2.13 at
  E=+0.05 on zgnr) is an unconstrained constant background whose
  sweeping KK ramp no constant H-correction absorbs; it biased the
  transmission by -0.17 around E=0. And `fit(..., eta_floor_mult=5.0)`
  - the width floor is 5x the data's imaginary broadening (5 eta =
  0.05 eV for zgnr), applied to BOTH stages of the two-stage
  dispatcher. Resolution-limit poles (2-3 eta) made the fit oscillate
  around the sampling function; with the floor equal to the true
  sharp-feature width the fit represents the feature with one
  properly-sized pole. Parameter sweep on zgnr (T vs the sampled-SE
  reference, E >= -3): mult=2 feature T 0.64, mult=3 0.73, mult=5
  0.815 (reference 0.819, old 3.py fit 0.81); oscillation metric
  0.0023 vs the old pipeline's 0.0021 - at parity, with T rms(E>=-3)
  0.170 vs the old fit's 0.195. Both knobs are defaults in
  `refine_sampling._fit_one` and the comparison scripts.
- **Pole-pair symmetry** (user question, verified numerically to
  1e-16): every (e, w) term is exactly the conjugate-symmetric pair
  z = e +/- iw; the dispersive KK partner used by Zandpack's SE
  construction comes from the same pair. No unpaired imaginary parts
  exist in the model, ever.
- **Residual-boost two-path fit (user recipe)**:
  `refine_sampling.fit_boosted(E, F, base_kw, n_res=8)` - after the
  adaptive PSD fit saturates, fit the *residual* (samples - fit) with
  a much SMALLER pole budget (n_res, PSD-forced, fixed count) and add
  the two models (CombinedFit: concatenated e/w/C, still shared-pole,
  injectable). The saturated first pass is stuck in its basin; the
  correction is smooth and low-rank, so a fresh small optimization
  fixes sample-level error and spurious structure the first pass
  cannot. Control experiment (zgnr): fixed-count fits pushed to 128
  poles give T rms 0.154 / osc 0.0037 / feature 0.779 - WORSE than the
  69-pole boosted model (0.147 / 0.0021 / 0.810) despite nearly double
  the poles, so the boost's gain is not pole count. The correction fit
  MUST be PSD-forced: an unconstrained residual fit reaches min
  eigenvalue -0.84 and the oscillations return. n_res=8 beats
  n_res=4. Also established: target_rms is NOT the pole-count knob -
  the adaptive loop saturates at the same ~53 poles for targets
  2e-3/5e-4/2e-4 (the "no longer pays" break dominates); fixed
  n_poles starts are simply grown back by the insertion pass.
  All three Data2 comparison scripts use fit_boosted(n_res=8).
- **Secondary plan round (archived secondaryplan.md in .junk/), three findings**:
  (a) **w_max_raw=0.35 beats 0.5** on zgnr: T rms 0.144 -> 0.130 at
  the same adaptive+boost recipe, sharp-feature 0.789 -> 0.815
  (reference 0.813). A fixed90 base + boost8 at 0.35 reaches 0.114
  (72 poles/lead - flagged as high). Fixed60 is NOT enough - the
  optimizer prunes it to 49 poles and T degrades to 0.248.
  (b) **Sampling-window extension** (the big win): the 2.py contour
  windows are NOT a constraint. Resampling zgnr on [-12, 12] (361
  pts, one extra TBtrans run, ~50 s) and fitting THAT moves the
  fit's uncontrolled edge decay far outside the range of interest:
  T rms vs the sampled-SE reference improves EVERYWHERE, including
  inside the old window - core [-3,3]: 0.136 -> 0.087, physical
  [-3,7]: 0.118 -> 0.070, and the old fit's edge blowup at +8.65
  (max dev 0.85) disappears (0.51, now at a band-edge cusp +2.66).
  Driver: `Data2/zgnr/extend_window.py` (uses refine_sampling.resample;
  wide runner pickled as tmp_wide.gz, fits cached in
  refined/wide_fits.npz). The zgnr comparison script now fits the
  wide window (old pipeline stays on the original 251-pt sampling).
  Final zgnr: OLD 3.py 0.189 vs NEW 0.070, sharp feature 0.813 vs
  0.812 exact (old 0.811). 61/65 poles per lead.
  (c) **Full-suite oscillation check**
  (`bench/oscillation_check.py`): every Data/ example fit with the
  full policy (PSD + width caps + boost); oscillation proxy = rms
  of (fit - PCHIP(samples)) on a dense in-window grid, normalized;
  figures + summary in results_lorentz_fit/oscillation_check/.
  Results: no oscillation plague anywhere (ripple ~ sample rms for
  all but aubj, whose sampling is known to be under-resolved).
  BUT `eta_floor_mult=5` is NOT universal: at eta=0.02 it forbids
  poles < 0.1 eV and destroyed fig2ag5/spinpol (rms 0.53-0.63 vs
  ~0.006 unconstrained) — their sharp features are narrower than
  5 eta. The multiplier is only safe when the sharpest feature is
  ~5 x eta wide (true for zgnr). Rule: mult=5 only for eta-limited
  data; mult 1-3 otherwise.
- **Why the width window [floor, w_max] controls the transmission
  oscillations** (measured evidence):
  UPPER limit (w_max ~0.35-0.5): a pole with a width comparable to
  the window is not a peak but an uncontrolled BACKGROUND - its
  absorptive part is a near-constant offset (which the fit happily
  uses to absorb sample-level mismatch) while its Hilbert partner is
  a window-spanning RAMP of the self-energy's REAL part. find_correction
  absorbs only a CONSTANT offset, so the ramp translates into an
  energy-DEPENDENT level shift -> slow T undulation across the
  window (measured on zgnr: a w=2.13 background pole biased T by
  -0.17 around E=0; capping w_max=0.5 removed it, 0.35 improved
  further). Wide poles also decay too slowly OUTSIDE the window, so
  the fitted SE does not fall off like the true one - capping keeps
  the out-of-window decay physical. Caveat: the cap must not cut
  below the widest REAL feature - zgnr_spin's broad spin-polarized
  features need 0.5 (0.35 regressed its T rms 0.216 -> 0.266).
  LOWER limit (~5 x eta / local sample gap): poles narrower than the
  data can support are INVISIBLE at the samples - the optimizer
  uses them to cancel residuals, and they ring violently between
  samples (measured on VerySmallRSSE: w=0.067 poles inside 0.13-0.2
  sample gaps spiked Tr Gamma by 25 between samples -> the T
  ringing in [3,5]). The robust cure is NOT a wider floor but a
  DENSER GRID where the ringing widths become legitimate.
- **VerySmallRSSE E=0 anti-resonance, solved** (two findings):
  (a) lead0's Gamma has a deep narrow NOTCH at E=0 (Tr Gamma
  48 -> 3 -> 48 over +-0.3 eV). PSD-FORCING fundamentally fills
  such notches: the convex PSD solve at any fixed pole set lands at
  Tr@0 ~ 19 (the notch shape requires near-cancelling positive
  terms, which the PSD projection smooths away). `psd_force=False`
  keeps the notch-fitting (unconstrained) model through the
  accept/reject logic, and with the DENSIFIED grid (0.04 spacing
  near 0) its padded-grid min eigenvalue is POSITIVE anyway
  (+0.016): the notch model is PSD-compatible without being forced.
  (b) The edge-band T ringing (see width writeup above) was cured by
  DENSIFYING [+-3, +-5] to 0.067 spacing + small wings to +-6
  (decay room; the artifact pole at the window edge disappears).
  Final RSSE recipe: 189-pt grid (E=0 cluster + dense edges +
  +-6 wings), psd=True psd_force=False, eta_floor_mult=3,
  w_max_raw=0.5, boost 8. T rms vs TBtrans: 0.090 (old pipeline)
  -> 0.0079, every band <= 0.01, T(0) 0.0212 vs 0.0237 exact.
  Driver: refined/improve_edges.py.
- **zgnr_spin window extension**: a UNIFORM wide grid (v1, 213 pts
  over [-12,12]) is WORSE than the narrow adaptively-refined grid -
  it undersamples exactly the features the refinement had resolved
  (core Trms 0.377 vs 0.216). The correct recipe keeps the refined
  grid and APPENDS sparse wings (v2: 91 refined pts + wings to
  +-12, 111 pts): full-window Trms 0.222 vs 0.436 (edge blowup at
  +8.65 eV gone) at a small core cost (0.219 vs 0.198). The
  adopted spin fit stays the narrow one (user-approved); v2 is
  pickled (tmp_wide2.gz + refined/wide2_fits.npz) for when the
  far-window behaviour matters.
- **Verbosity**: `lorentz_fit.fit(..., verbose=True)` now narrates
  the whole pipeline (start config -> init -> prune/add rounds ->
  polish -> duplicate merge -> local-floor widening -> PSD
  accept/reject with both rms values -> one-line final summary),
  and the two-stage floor dispatch reports which stage won.
  `refine_sampling.refine_loop` (verbose=True default) prints the
  fit kwargs, per-(lead, ik) fit results WITH the proposed
  insertion energies, per-round grid growth, per-lead removal
  candidates, and the final fits.

Measured effect (zgnr refined grid, psd_force): Gamma rms
0.43/0.28 -> 0.021/0.021 with FEWER poles (23/28 vs 47/37); on the
review's own synthetic benchmark the PSD-forced fit improved from
0.132 (their pipeline measurement) to 0.052 with dense-grid error
0.23 -> 0.146. Not yet implemented from the review: dense-grid
validation for model selection (4.2, validation-only variant),
edge-continuation anchors (4.5), factored L L* joint optimization
(4.6c), committee-driven refinement (4.11).

### 4.10 TL;DR — drop-in recipe for a 3.py script

NOTE (2026-10-08 evening): the active implementation is now the
single file `AI_fit.py`; `refine_sampling.py` is a shim re-exporting
it, so all imports below keep working unchanged (new code can
`import AI_fit` directly). During the switch, three merge bugs were
found and fixed in AI_fit.py's `inject_fits` (`Gvals`/`Ginds` ->
`G_samp.vals`/`G_samp.inds`, `GBlock_shape` -> `G_samp.Block_shape`)
— found by the nogap_Ag5 end-to-end run + an AST undefined-name scan
+ a token-level diff of every function vs the checkpoint original;
the package-era implementation (checkpoint folder) is otherwise
reproduced exactly (only intentional `_fit` wrapper -> `fit`).

Five edits. Everything below the injection is your existing 3.py tail,
unchanged.

**Edit 0 — loads stay first (shadow rule).** The TD_Transport / Device
pickles must be loaded before the FittingRoutine root ever touches
`sys.path` (the repo-root Zandpack/siesta_python folders are
inspection copies only and shadow the installed packages):

```python
import pickle, gzip, sys, os
R = pickle.load(gzip.open('tmp.gz','rb'))
D = pickle.load(open('Device.SiP','rb')); D.mpi = "mpirun "
# ... orbs selection exactly as in 2.py ...
```

**Edit 1 — import the driver** (temporary root insert, removed at once):

```python
ROOT = <path to FittingRoutine>
sys.path.insert(0, ROOT); import refine_sampling; sys.path.remove(ROOT)
```

**Edit 2 — OPTIONAL: refine the sampling grid.** One call replaces
2.py's fixed contour going forward. It reruns TBtrans per round
(adds points where under-resolved, drops redundant ones), and the
sampling stored in R is the refined grid afterwards:

```python
out = refine_sampling.refine_loop(
    R, D,
    read_kwargs=dict(sub_orbital=orbs, less_memory=False),  # as 2.py
    n_rounds=3, insert_tol=5e-3, remove_tol=1e-2, do_remove=True)
R.pickle('tmp_refined', compression='gzip')   # keep it
```
For expensive per-round fits add `derivative_only=True` (a fit-free
pre-pass: insertions come purely from d(sum_ij |F_ij|)/dE, see 4.8).
Spin systems (zgnr_spin-style): add `runner=lambda R_, D_:
D_.mimic_tbtrans_spin(R_.Contour), eta=2e-2` and run the folder's
`sample.py` once first. RSSE systems: use the rebuild-folder pattern
of `Data2/VerySmallRSSE/refined/` (see 4.8) — the in-place loop does
not apply. The COMPLETE refinement recipe — when to refine at all, the
full knob table, per-system-type patterns (TBtrans / spin / RSSE), the
derivative-only pre-pass, grid verification, and the wall cost — is
section 6 of `GUIDE_3py_integration.md`.

**Edit 3 — the fit.** `out['fits']` already holds the per-(lead, k)
FitResults. The workflow default is psd=True + psd_force=True (the PSD
model is accepted even at rms cost: PSD residues make the assembled
Gamma ≥ 0 everywhere — between samples and beyond the window — which
the transport quantities require; an unconstrained Γ tail can go
negative, producing unphysical self-energies). EXCEPTION: if the
sampled Gamma has a deep narrow NOTCH (dips toward zero), use
`psd_force=False` — PSD-forcing fundamentally fills notches (4.9,
VerySmallRSSE). If you skipped Edit 2, fit directly — `fit_boosted`
(the adaptive PSD fit + a small PSD correction fit of the residual)
measurably beats both the plain fit and any fixed pole count (4.9):

```python
fits = {}
for lead in range(R.num_leads):
    for ik in range(len(R.Nonortho_Gammas[0].vals[0])):
        E, F = refine_sampling.extract_dense(R, lead, ik)
        fits[(lead, ik)] = refine_sampling.fit_boosted(
            E, F,
            dict(n_init=24, max_poles=100, target_rms=2e-3,
                 psd=True, psd_force=True,
                 eta_raw=1e-2,          # = your contour's imaginary part
                 eta_floor_mult=5.0,    # SYSTEM-DEPENDENT: 5 for zgnr
                                       # and the metals, 1 for agnr and
                                       # the eta=0.02 systems (4.9/6.3)
                 w_max_raw=0.5,         # width ceiling (4.9); 0.35 only
                                       # if slow T undulation persists
                 local_floor_frac=1.5,
                 channel_tol=1e-4 if F.shape[1] > 32 else 0.0),
            n_res=8)
```

The knob values above are generic starting points; the per-system
tuned table (eta_floor_mult, w_max_raw, psd_force, pole caps) is
6.4 — measure before trusting defaults. Solver note (2026-10-09):
`tr_solver` defaults to **'lsmr'** (faster, smoother residuals,
better notch fidelity; its steep-band-edge error is a same-sign BIAS
that accumulates in integrated quantities). Pass `tr_solver='exact'`
in the fit dict when integrated currents across band edges are the
target observable, or when a bit-for-bit comparison against
pre-2026-10-09 stored results is required (see 4.11 for the measured
tradeoffs and figures). A third solver `tr_solver='gram'` (with
`gram_cut=1e-9`) is the fastest option and pairs with a MULTI-START
HEALTH GATE on basin-sensitive systems (Ag3Au3 measured; 4.13/4.14:
judge by padded min eigenvalue AND sample rms together, best healthy
of 3 jittered starts). The complete, copy-paste-ready 3.py template —
including the multi-start/health-gate helpers, the solver table, the
verification snippet, and the pitfall checklist — is
`GUIDE_3py_integration.md` at the repo root (smoke-tested 2026-10-09
on Ag3Au3 lead0).

**Edit 4 — inject into the Zandpack containers.** This single call
replaces the old PoleGuess → R.Fit → curvefit_all → iterative_PSD →
FitNO2O block (the injected model is the SAME sum of shared-pole
Lorentzians the Zandpack machinery already evaluates — identical pole
convention w²/((E-e)²+w²) and per-ik pole arrays):

```python
refine_sampling.inject_fits(R, fits)   # also runs FitNO2O
```

Multi-k-point systems (zgnr_spin/spinpol-style, n_ik > 1): nothing
extra to do — `inject_fits` GUARANTEES one common pole count for all
(lead, ik) sets (pads to the global max with inert zero-residue
poles; Zandpack's R.NumL helpers require it — 4.7.2). If you want the
fits THEMSELVES equalized (e.g. before pickling), call
`refine_sampling.equalize_pole_counts(fits)` first (mode 'dup' or
'zero'; purely additive, evaluate() unchanged).

**Then keep your 3.py tail exactly as it is:**

```python
from Zandpack.FittingTools import find_correction
C = find_correction(R, Emin=-3, Emax=3.0)
R.Renormalise_H(C)
R.tofile('src', tol=1e-3, write_ixi=False)
R.Inspect_transmission_from_hilbert_transform(np.linspace(-11,11,550),
                                              eta=1e-2, figure="Tfit.svg")
```

Reference implementations: `Data2/zgnr/compare_dos_t.py` (old pipeline
and new pipeline side by side), `Data2/zgnr_spin/compare_dos_t.py`,
`Data2/VerySmallRSSE/refined/compare_dos_t.py`.
Verification harnesses: `tests/` (the ported ALL PASS suites),
`validation/` (65 gate/benchmark/diagnostic scripts collected from
the /tmp scratch, README-documented, one gate per optimization
round), `Data2/VerySmallRSSE/profiling/` (profile + micro-bench),
and the per-system figure scripts living next to their data.

### 4.11 Performance (guides archived in .junk/; measured outcomes in
`performance_guides_outcome.md`)

Profiled at p=64, N=451, psd (459 s baseline; cProfile). Hotspots were
the TRF/LSMR pole relocation (74%, cost ~ n_ch) and the PSD path (25%,
dominated by a per-pole eigh Python loop). Implemented and verified:

- **Channel compression** (guide 4.1, the big one): the response
  matrix B (n_ch, N) is numerically LOW RANK for smooth level-width
  data (measured on real Ag3Au3: rank 9-19 of 729 channels at
  1e-9-1e-4 relative). The LS cost is invariant under orthogonal
  channel rotation and the discarded directions carry a
  theta-independent residual, so the optimizer trajectory is UNCHANGED;
  `fit_poles` runs on the compressed rows and the final coefficient
  solve is always against the FULL responses (exact X/R).
  `compress_channels(B, rank_tol=1e-9)`; compressed ONCE per
  `_fit_impl` and passed as `B_opt` to every internal `fit_poles`
  call (guide 2.1).
- **Factor cache** (4.3): scipy calls fun(theta) then jac(theta) at
  the same point; one-entry memo. Exact.
- **Vectorized dense Jacobian** (4.2) and the `basis_apply` matmul
  replacing every `einsum('ni,ijk->njk')` (4.8) - exact (allclose
  verified for every rewrite, plus a pack/unpack/project_psd
  equivalence test).
- **PSD path**: batched (stacked-LAPACK) `project_psd`, vectorized
  ADMM channel pack/unpack, one Cholesky factor for all ADMM
  iterations, ADMM OVER-RELAXATION alpha=1.6 with tolerance 1e-6
  (guide 2.3; verified: 200 iterations reach exactly the 5000-iteration
  reference objective), and the per-candidate RANKING psd_refine uses
  admm_iters=50 (the final refine keeps the full budget).
- **Loose intermediate tolerances** (guide 2.2): the init/prune/
  insert/merge/local-floor refits run at ftol=xtol=gtol=1e-6 - they
  only seed the next candidate; the final polish keeps 1e-8.

Measured: profile case 459 s -> 53 s (8.7x); real Ag3Au3 lead1
~1900 s -> 782 s at the mid-point of the work (final code faster
still); test suite unchanged and ~20% faster. NOT adopted: Gram-matrix
LM replacing TRF (2.4, invasive), parallel floor stages (2.5, wastes
CPU), partial refits (2.6), numba basis kernel (available - numba
0.67 on this Python - but post-compression the arrays are small; only
worth it if a re-profile shows per-call NumPy overhead dominating).

**Threading**: keep `OMP_NUM_THREADS=1`. Measured on the real Ag3Au3
fit: 8 threads gave 1068 s vs 782 s at 1 thread - post-compression
the matrices are too small to amortize thread overhead, and threaded
BLAS breaks the run-to-run reproducibility (see caveat 1). Re-measured
2026-10-09 on the fully optimized gram pipeline (2 threads vs 1,
production test): 226 s vs 204-193 s per 6-fit run - ~11% SLOWER at
n=2 as well, and EVERY start landed a different basin (threaded BLAS
reduction order; the health gate still produced a healthy pair, T rms
0.009385 vs 0.007719). ENVIRONMENT NOTE: the ambient shell exports
OMP_NUM_THREADS=1 (so `nproc` reports 1) while 18 cores are actually
available - do not mistake the nproc output for the machine's core
count (validation/11_gram_basin_spread/ag3au3_gram_production_omp2.*).

**RSSE re-profile (2026-10-08, Data2/VerySmallRSSE/profiling/)**:
N=189, p=6, n_ch=36 compressed to rank 12 IN PRODUCTION (compression
is already active everywhere via `B_opt=Bc_fit` — do not re-add).
Production fits: lead0 31 s / lead1 21 s. Remaining hotspots inside
`fit_poles` (95% of wall): scipy trf's per-iteration dense SVD of J
(56% lead0 / 33% lead1 — intrinsic to trf+tr_solver='exact'), the
`_varproj_factors` QR stack (16-43%), `residual_and_jacobian` (16-20%).
`fit_poles` scales ~M^2.4 (63 ms @ M=24, 415 ms @ 50, 1.56 s @ 100).
A/B test of forcing the matrix-free `VarProjJacobian` + lsmr path on
RSSE (`operator_threshold=0`): ~1.7-2.2x faster per solve BUT FAILED
the quality gate — adaptive-loop rms regressed 4.9x on lead1 and PSD
broke (minEig -2.0 / -6.4e-2; the notch accept-guards land in bad
basins). NOT adopted; the operator path stays threshold-gated
(`fit(..., operator_threshold=3_000_000)`, knob threaded through all
internal `fit_poles` calls, default verified bit-for-bit identical to
the pre-knob module on 6 synthetic adaptive fits + ported tests).
Realistic remaining levers: cache/warm-start `_varproj_factors` QRs
across line-search steps, or reduce refit counts in the adaptive loop.
NOTE: `refined/cmp_fits.npz` does NOT correspond to the current
189-pt `C1.xz` (43/81-pole fits, likely from the pre-densification
grid) — regenerate before trusting injection comparisons.

**`_varproj_factors` allocation cleanup (2026-10-09, user review)**:
`solve_triangular` import moved to module level (was re-executed
inside the try block on every call); intermediate temporaries
eliminated with in-place ops (`Res = A@X; Res -= Y`,
`np.subtract(D, Z, out=Z)`, `np.multiply(D, sw, out=D)`; bitwise
identical, incl. the rank-deficient fallback path — 15-case unit
identity + 6-fit Gate A + ported tests). Measured effect: NONE
(0.95-1.01x per call) — the function is LAPACK-bound (QR + triangular
solves), allocation overhead is ~1-3%. Persistent preallocation was
investigated and REJECTED: all five outputs are returned and
retained (factor cache between fun/jac, VarProjJacobian across
iterations), so buffer reuse would need copy-out, negating the gain.

**Full-systems benchmark agreement (2026-10-09)**: all 10 Data2
systems rerun end-to-end through the current single-file AI_fit
(via apply_method + compare_dos_t) reproduce every stored result —
7/7 method_summary fields bit-level (only machine-epsilon min_eig
~-1e-16 differs on agnr/Fig2Ag5), zgnr 0.1893/0.0698, zgnr_spin
0.4541/0.2159, RSSE 0.08996/0.00796. Fixes made during the rerun:
band-table guard for windows not covering a band (agnr twin [-3,7]),
and zgnr/compare_dos_t.py legacy `from lorentz_fit import validate`
(after-root-removal import that only worked when package-era
refine_sampling preloaded lorentz_fit into sys.modules) → now
`import AI_fit as val`. Stored summaries snapshotted as
`method_summary_stored.json` beside the fresh ones.

**`_varproj_factors` guide round (2026-10-09, varproj_factors_speedup_guide.md)**:
MEASURED first: 52.3% of `_varproj_factors` calls in a production RSSE
fit are cost-only (scipy trial steps — the guide's A1 target); SVD
fallback fires only ~30x/fit (B1 not needed). ADOPTED: A1 lazy
core/Jacobian split (`_varproj_core`/`_varproj_D`/`_varproj_jac`;
trial steps skip D/Z/K/S; `fit_poles`'s dense-jac branch now uses the
cached factors via `_assemble_dense_j` instead of recomputing
`_varproj_factors` inside `residual_and_jacobian` — the old path did
the QR block TWICE per accepted step) + A3
(`check_finite=False` on the K triangular solve). Bit-identical
(8-case unit incl. fallback paths + Gate A 6/6 exact + RSSE end-to-end
n/rms identical); ~8% end-to-end RSSE (25.0 -> 23.0 s; scipy's
internal trust-region SVD dominates and is untouched). Also: AI_fit
now falls back to the numpy J path when `NUMBA_DISABLE_JIT=1`
(previously the uncompiled pure-Python kernel would have been used).
REJECTED with reasons: A6 unpadded ridge products (fails the exact
gate, worst 8e-13, and 0.91-1.07x — no gain); A2 (subsumed by A1);
A4 (OMP=1 already measured faster); B1/B2/B3 (fallback rare / would
relax the bit-for-bit gate); reflector-QR-without-Q (guide's flop
count: ~1.3-1.6x MORE flops for our many-RHS pattern — the earlier
20-40% estimate was wrong). Numba traps: `cache=True` kernels cannot
be loaded from dynamically-named modules ('<dynamic>' ModuleNotFound
— stale cache indexes in backups/__pycache__ must be deleted); gate
harnesses force the backup's `_njit = None` for the numpy path.

**speedupguide_new.md round (2026-10-09)**:
- Step 1 profile of the 23s RSSE fit: scipy trust-region SVD 66.5%,
  `_varproj_core` 0.8%, `_varproj_jac`+J assembly 3.0% — the varproj
  work is done; remaining time is scipy-internal.
- Step 2 nogap memory A/B (one fit(), same inputs, identical result):
  pre-A1 peak 901 MB vs current 819 MB — growth is PRE-EXISTING
  (staircase climb to a plateau = adaptive working set growing with
  pole count + glibc arena retention; not a leak, not A1-caused; A1
  actually lowered the peak). NOGAP BENCHMARK UNSHELVED.
- Step 3 cleanups ADOPTED (bit-identical: 8-case unit + Gate A 6/6 +
  tests): QR-path core drops A/Y (unread), theta.copy() stored, D
  built in one pass into a preallocated (N+M, 2M) array.
- Step 4: already done — compression verified active (36->12 channels
  on RSSE; the SVD runs on the compressed J).
- Step 5 tr_solver: **lsmr is now the DEFAULT** (2026-10-09, user
  decision after the bias/ripple analysis; `fit(..., tr_solver=...)`,
  default "lsmr", opt-out with 'exact'). Validated: ported tests ALL
  PASS under the default; operator branch (LinearOperator jac)
  accepts explicit lsmr; RSSE fit() default gives n=59 rms 5.1e-4
  (exact path: n=42 rms 5.9e-3).
  MEASURED, per method (RSSE production, eta-matched T gates):
  - lsmr: 1.55x faster RSSE (22.9->14.8 s), 1.76x nogap lead0
    (659->374 s); Gamma rms 11x better on RSSE (0.0114->0.00103);
    T(0) notch better (0.0214 vs ref 0.0224; exact 0.0240);
    residuals smooth (ringing_rms 0.0002).
  - exact: near-zero-mean ripple (bias_mean -0.0007) that averages
    out in integrals (band-integral rel err -0.11% vs lsmr +2.09%).
  CAVEATS — lsmr: (1) steep-band-edge error is a same-sign BIAS
  (bias_mean +0.0139, bias_max 0.0347 in RSSE [3,5]) that
  ACCUMULATES in integrated observables (currents) — pass
  tr_solver='exact' when integrated currents across band edges are
  the target; (2) 1.56x slower than exact for cold small-M solves
  (below M~40; adaptive loops still net faster); (3) all pre-2026-10-09
  stored results (cmp_fits, summaries) were produced with exact —
  regenerate before mixing.
  CAVEATS — exact: (1) 1.5-1.8x slower for M >~ 40 (per-iteration
  dense SVD of J); (2) its residual is a small oscillation
  (ringing_rms ~0.001, zero-mean) — harmless for integrals, spurious
  for dT/dE-type observables; (3) worse notch fidelity on RSSE.
  Evidence figures: Data2/VerySmallRSSE/refined/lsmr_edge_ringing.png
  (bias wording), Data2/nogap_Ag5/lsmr_tradeoff.png; split-metric
  harness validation/05_lsmr/verify_bias_vs_ringing.py (savgol bias/ripple split,
  int_rel_err); doc: lsmr_vs_exact_edge_band.md (correction of the
  earlier 'ringing' mislabel: lsmr's [3,5] error is 100% same-sign
  smooth bias, NOT oscillation).
  nogap FULL T-level A/B (make_lsmr_tradeoff_figure.py; NOTE both
  modes ran with channel_tol=1e-4 — internally consistent A/B, but
  its exact baseline differs from the production cmp_fits which use
  channel_tol=0.0 at p=27): exact 0.0603 overall vs lsmr 0.0707;
  lsmr worse in most bands, worst the plateau [-5,-3) 0.039->0.058;
  equal in [-2,2) and [4,6); lsmr 1.49x faster (855 vs 1272 s, both
  leads) with fewer poles (51/65 vs 70/70). If nogap T fidelity
  matters, pass tr_solver='exact' for that system.
- Step 6 (cost-only reflector path): skipped — the profile shows the
  core at 0.8% of wall; there is nothing left to win there.

**`never_form_J` guide round (2026-10-09, tr_solver='gram')**: a
vendored copy of scipy 1.18.1's `trf_bounds` (`AI_fit._trf_gram`) in
which the optimizer NEVER sees a Jacobian — the callback returns
`(G, g) = (J^T J, J^T f)` built directly from the varproj factors
(`AI_fit.gram_and_grad`; the Z^T K = 0 orthogonality kills the
cross terms, so G is two Hadamard products of small matrices, cost
INDEPENDENT of the channel count). The per-step SVD of the augmented
Jacobian is replaced by one `np.linalg.eigh` of the augmented Gram
matrix; scipy's `select_step`/quadratics run in Gram form
(`_gram_select_step`); `fit_poles(..., tr_solver="gram",
gram_cut=1e-9)`; the K triangular solve is skipped on the QR path
(`_varproj_jac(..., with_k=False)`); `np.linalg.eigh` (dsyevd) per
the guide's driver measurement. Uses scipy PRIVATE API
(`scipy.optimize._lsq.common`) behind a guarded import — rerun the
regression guards (`tests/AI_fit_test_synthetic.py::check_gram_solver`,
validation/10_gram_solver/) after any scipy upgrade.
VALIDATION (validation/10_gram_solver/, guide section 5 in order):
- identities G=J^T J / g=J^T f: 4e-16..5e-15 on synthetic
  RSSE/ag3au3/nogap shapes AND real RSSE (early + late iterates),
  QR path AND forced SVD fallback; eig(G) == sv(J)^2 to 1e-14
  (one 1.8e-8 outlier at cond(J)=5e15 — exactly the sqrt(eps)
  eigh floor the guide predicts); select_step agreement 4e-19.
- trust-region step vs scipy's exact SVD path (Delta 0.05-5x GN,
  active bounds, both factor paths, real RSSE): worst 2.5e-14
  (guide's own synthetic worst: 5e-9 at cond 4e4).
- trajectories: RSSE lead1 gram follows 'exact' EXACTLY (nfev 194=194,
  cost to 8 digits, poles to 1e-10); lead0 (max_nfev-capped, still
  descending) gram reaches slightly DEEPER cost at poles within 2e-2.
  Synthetic gate: gram == exact minimum to 4.8e-14 relative cost.
- SPEED (RSSE production fit_boosted, medians of 3): exact 27.3 s /
  lsmr 18.6 s / gram 9.2 s (rms 0.0114 / 0.00103 / 0.0085 — the
  adaptive loop is basin-sensitive for EVERY solver, same phenomenon
  as the lsmr-vs-exact round). fit_poles-level (n=59 cold): gram
  0.82 s vs exact 6.67 s (8x) and lsmr 5.22 s.
- Ag3Au3 (the system PINNED to exact because lsmr lands non-PSD
  basins): exact 342.9 s / rms 0.0149 / minEig -1.7e-2; gram
  45.6 s (7.6x) / 0.0228 / -3.2e-2 at gram_cut=1e-9. Identity
  check on real ag3au3 factors: 6.5e-16, cond(J) 1.4e4, zero
  truncated directions — the fit_poles math is faithful; the
  production difference is ADAPTIVE-LOOP BASIN DIVERGENCE.
  gram_cut sweep: 1e-7 (the guide's suggested CUT) wrongly
  truncates a genuinely weak direction on this system (rms 0.0276,
  minEig -1.6e5); 1e-9 and 1e-11 are bit-identical and healthy ->
  DEFAULT gram_cut = 1e-9 (measured; see _fit_impl docstring).
DECISION: the DEFAULT solver stays 'lsmr' (user-set 2026-10-09);
'gram' is opt-in. Ag3Au3 / Ag3Au3_gap were PINNED to 'exact' on the
single-start comparison below — SUPERSEDED by the basin spread
(next paragraph + 4.13): the pin is unsupported (gram@1e-9 healthier
in distribution and ~5x faster); user decision pending (section 8).
For every other system 'gram' is the fastest option and tracks
'exact' closely (use it when fit time dominates, e.g. large-M
refits inside refine_loop rounds). Consolidated gram reference:
4.14.

**Perturbed-start basin-spread round (2026-10-09, validation/
11_gram_basin_spread/)**: 7 starts (control + 6 x ~1e-6 eV init-pole
jitter) x 4 solvers on Ag3Au3 lead1 and VerySmallRSSE lead0, each
run measured at Gamma level AND T level (injection + find_correction
+ Hilbert-transform inspection vs the reference). Findings:
(a) gram cut 1e-9 == cut 0 bit-identically in all 56 runs — the
1e-9 truncation never fires on real systems; 1e-7 is actively
harmful on Ag3Au3 (1/7 healthy vs 4/7) -> gram_cut default 1e-9
CONFIRMED; (b) exact's OWN start-to-start spread is as bad as or
worse than gram's — on Ag3Au3 the exact CONTROL start (the
production configuration) lands the worst basin of the study
(rms 0.0258, padded min eig -236, T rms 3.44 with T_ref max 1.53),
so the single-start comparisons behind the Ag3Au3 'exact' pin
(0.0228 vs 0.0149) were inside basin noise; gram@1e-9 has the
better T-rms median (0.014 vs 0.020) and health rate (4/7 vs 2-3/7)
at 4.7x lower wall; on RSSE gram is 6/7 healthy vs exact 5/7 at
3.6x lower wall; (c) HEALTH GATE (user-directed): judge fits by
padded min eig AND T-level error TOGETHER, never by sample rms alone
(measured: a PSD model at rms 0.105 / T 0.025 AND a min-eig -236
model at rms 0.026 / T 3.44 on the same system) — production recipe
for basin-sensitive systems: 2-3 jittered starts, keep the fit that
passes; (d) era forensics: the Ag3Au3 exactera cache is
bit-consistent with a fit on TODAY's data, same width fingerprints
(wmin 0.05000 / wmax 0.50000), and every code era (package
checkpoint / pre_channelcomp / current, numba on/off) reproduces
the current exact fit bit-identically — the cache is one sample of
the same chaotic basin distribution, nothing is stale or broken,
and its stored T 0.0100 is beatable (spread best T 0.0075 at rel
rms 0.0045). METRIC TRAP: frobenius_rms_error is RELATIVE —
comparing it to an absolute rms produces bogus conclusions (bit us
once during the forensics). PRODUCTION TEST of the resulting recipe
(gram@1e-9, 3 starts/lead, health-gated selection, full
post-processing, Ag3Au3): T rms 0.00937 over (-5,9) — beats the
stored exact-era pair (0.0100) and the old pipeline (0.0383) at
~6x lower fit wall than the exact-solver equivalent; the gate
rejected lead0's unhealthy control start (min eig -1.38) as
designed (validation/11_gram_basin_spread/ag3au3_gram_production.*).

### 4.12 Dependency guide — what affects what

For agents picking up the code: the causal map from inputs and knobs
to behavior. Evidence pointers are the sections/tables above.

DATA (from 2.py / refine_loop) — affects everything downstream:

| input | affects |
|---|---|
| E grid (non-uniform OK) | w_min floor (median spacing), quadrature weights, insertion/removal proposals, skrf init |
| eta (contour imaginary part, via eta_raw) | HARD width floor (eta_floor_mult x eta): structure narrower than eta cannot exist in the data; also the comparison eta must MATCH the sampling eta (the nogap 0.355-vs-0.031 artifact) |
| F samples | channel rank (compression), sample weights, PSD feasibility (deep notches) |

FIT KNOBS — what each one moves:

| knob | affects | evidence |
|---|---|---|
| eta_floor_mult | width floor: too low -> between-sample oscillation; too high -> clips sharp features (can collapse the adaptive loop) | zgnr 5 vs agnr 1 (mult 5 catastrophic); nogap mult 2/3 collapse to 15/13 poles |
| w_max_raw | forbids window-spanning broad poles (unconstrained constant background + sweeping KK ramp that find_correction cannot absorb) | nogap: 0.3 beats 0.5 (T 0.068 vs 0.080) |
| local_floor_frac | width floor relative to LOCAL sample spacing (sparse regions of refined grids) | non-uniform stress tests |
| psd / psd_force | PSD Gamma >= 0 everywhere vs accepting non-PSD models when they fit samples better; psd_force=True FILLS notches | RSSE E=0 notch (4.9); Ag3Au3 non-PSD basin under lsmr (6.3) |
| target_rms / max_poles | adaptive stopping; runtime ~ M^2.4 per solve | nogap pole-cap sweep (70 = 150 quality) |
| n_res (residual boost) | small second PSD fit of the residual | nogap: 12 worse than 8 |
| channel_tol | drops noise channels from BOTH the optimizer and the final solve (their coefficients are zeroed) | large-p systems |
| tr_solver | 'exact': slow, zero-mean ripple, bit-reproducible per start; 'lsmr' (default): 1.5-1.8x faster, smooth bias at band edges, non-PSD basins possible under psd_force=False; 'gram': fastest (2-8x vs exact), never forms J (channel-count-independent) — per the basin spread (4.11/4.13) ALL solvers are equally basin-sensitive on Ag3Au3 (exact's own control start was the worst basin measured), so solver choice is a SPEED decision once the multi-start health gate is in place; gram_cut=1e-9 default (= cut 0 in practice; 1e-7 harmful on Ag3Au3) | 4.11 spread paragraph; 4.13; validation/10_gram_solver; validation/11_gram_basin_spread |
| ridge | tames coincident/sub-resolution poles (the astronomically-cancelling-coefficient spike mechanism) | solve_coefficients docstring |
| weight='relative' | per-sample 1/max(||F_n||, eps): peak fidelity vs global-scale fit | _sample_weights docstring |
| init / n_init | basin selection: skrf relocation / NNLS PSD dictionary / linspace | psd=True routes to the dictionary |
| tail_anchors | low-weight zero targets beyond the window edges -> decay behavior | fit() docstring |
| e_margin | how far pole centers may sit outside the window |

PIPELINE — who consumes what:

```
E, F -> prepare (scale) -> hermitian_channels/pack (n_ch = p^2 real)
      -> channel_tol drop -> compress_channels (rank r, FIXED per fit)
      -> init poles -> _fit_impl adaptive loop
             fit_poles: _varproj_core (QR/X/Res) + lazy _varproj_jac
                        (D/Z/K/S), trf via tr_solver, one-entry caches
             psd_refine / psd_solve_admm (accept-guards rank by
                 PSD-projected rms)
      -> FitResult -> fit_boosted (CombinedFit = fit + residual fit)
      -> [equalize_pole_counts]  (optional, exact)
      -> inject_fits (equal_nl='all' pads to the global max;
                      setLval + FitNO2O)  -> R.NumL contract
      -> find_correction + Renormalise_H (Zandpack, window-matched)
      -> transport quantities
refine_loop (4.8) changes E/F upstream -> everything re-fits
```

ENVIRONMENT: numpy + scipy required; numba OPTIONAL (dense-J kernel,
bitwise-identical numpy fallback, also under NUMBA_DISABLE_JIT=1);
OMP_NUM_THREADS=1 (measured faster + reproducible; threading changes
BLAS reduction order).

### 4.13 Basin-spread round: findings, the confusion, and the traps
(2026-10-09; validation/11_gram_basin_spread/ with its README; the
compact results summary lives in the 4.11 spread paragraph)

Purpose (project_handoff point 3): decide gram_cut from DISTRIBUTIONS
instead of single starts. 7 starts (control + 6 x ~1e-6 eV init-pole
jitters, fixed seed) x 4 solvers (exact; gram cut 1e-7/1e-9/0) on
Ag3Au3 lead1 + VerySmallRSSE lead0, every run measured at Gamma level
AND T level (fresh pickle, candidate lead injected with the other
lead fixed, find_correction, Inspect_transmission_from_hilbert_
transform vs the sampled-SE / TBtrans reference). Outcome: gram_cut
1e-9 confirmed (== cut 0 in all 56 runs; 1e-7 harmful on Ag3Au3),
the Ag3Au3 'exact' pin unsupported (exact's own control start landed
the WORST basin of the study: rms 0.0258, min eig -236, T rms 3.44),
health gate adopted (padded min eig AND T error TOGETHER), and the
production test of the recipe (gram@1e-9 + 3 starts/lead + gate)
reached T rms 0.00937 on Ag3Au3 - better than the stored exact-era
0.0100 and the old pipeline 0.0383 at ~6x lower fit wall.

What CONFUSED us, in the order it unfolded (recorded so the next
agent does not re-walk this path):

1. The control mismatch looked like a CODE REGRESSION. The
   unjittered exact control gave rms 0.0258/108 instead of the
   exactera cache's 0.0171/94 - the obvious suspects were the
   post-package performance rounds (channel compression, the numba
   J kernel, the A1 split) silently breaking the bit-identity
   claims. Each was tested with a full production fit and ELIMINATED:
   seeded-init_poles vs the natural unseeded run = bit-identical
   (the harness was fine); NUMBA_DISABLE_JIT=1 = identical (kernel
   innocent); the pre_channelcomp backup = identical (compression
   innocent); the package-era checkpoint code (the era that CREATED
   the cache) = identical (the merge innocent); an skrf-init instead
   of the psd dictionary = yet another basin (0.0487/65, not the
   cache). NO code era differs: the whole lineage is bit-reproducible
   on the exact path - in itself a strong validation of 4.11.
2. The data hypothesis, and the METRIC ARTIFACT that faked it. The
   cache predates the 4.7.1 block-storage fix, so different input data
   was the next suspect - but a per-block analysis showed Ag3Au3's
   off-diagonal blocks are NEGLIGIBLE (mean |F| 2e-5 vs 0.055 in the
   (1,1) block), so halving conventions cannot matter. Then
   re-evaluating the cache against today's extract with an ABSOLUTE
   rms produced "the cache fits today's data 7x better than its own
   stored rms" - an ARTIFACT: frobenius_rms_error is RELATIVE
   (sqrt(sum|diff|^2 / sum|F|^2)). Recomputed with the relative
   metric: bit-identical (lead0), 8 digits (lead1). The cache was
   fit on exactly today's data.
3. The config fingerprint closed the case. The cached widths sit
   exactly at today's floor and ceiling (wmin 0.05000 = 5 x eta,
   wmax 0.50000) - same knobs. With data, knobs, and code all
   identical, the cached fit is simply ONE SAMPLE of the chaotic
   basin distribution. The things we thought we knew from single
   starts - the gram-vs-exact 0.0228-vs-0.0149 gate that pinned
   Ag3Au3, the lsmr "regression" call - were inside basin noise.
   Nothing was ever stale or broken; what was wrong was our
   habit of comparing single starts on basin-sensitive systems.

TRAPS for the next agent (new + reinforced):

- frobenius_rms_error - and therefore FitResult.rms / comb.rms -
  is RELATIVE. Never compare it against an absolute rms; it faked a
  "data changed" conclusion here.
- Cached fit objects (cmp_fits*.npz) unpickle as module
  'refine_sampling' (the historical shim): import the shim (repo
  root temporarily on sys.path, then REMOVE it) BEFORE
  np.load(..., allow_pickle=True), or the load dies with
  ModuleNotFoundError.
- The repo-root shadow trap ALSO fires when python runs with CWD =
  the repo root (unpickling imports ./Zandpack and dies on its
  missing 'funcs' dependency). Run diagnostic scripts from /tmp or
  the validation folder, or os.chdir('/tmp') first.
- psd_dict_init and skrf_init are deterministic: passing
  fit(init_poles=(dispatch poles)) reproduces the natural unseeded
  fit BIT-IDENTICALLY - the sanctioned way to create controlled
  starts (the spread harness relies on it and validated it).
- Single-start solver comparisons on basin-sensitive systems carry
  no information. Use the spread harness + the joint health gate;
  note that sample rms alone accepts BOTH a PSD rms-0.105 model
  (T 0.025) and a min-eig -236 rms-0.026 model (T 3.44).
- The machine is ONE core: never run two timed runs concurrently;
  the spread runners resume from their JSONLs (delete to rerun from
  scratch). Full Ag3Au3 spread ~2 h; RSSE ~8 min; the gram
  production test ~12 min.

### 4.14 The Gram solver (tr_solver="gram") — consolidated reference
(absorbed from project_handoff.md, retired 2026-10-09; everything
below measured with OMP_NUM_THREADS=1. Full scripts + logs:
validation/10_gram_solver/ with its README.)

Implementation (AI_fit_base.py, was AI_fit.py; backup backups/AI_fit_2026-10-09_pre_gram.py):

- `gram_and_grad(Res, X, Z, S, R1/Q | K)` — G = J^T J (2M x 2M) and
  g = J^T f built directly from the varproj factors. The
  orthogonality Z^T K = 0 (range(A)^perp vs range(A)) kills the
  cross terms, so G is two Hadamard products — the cost is
  INDEPENDENT of the channel count. QR path: K never formed
  (K^T K = R1^-1 R1^-T); SVD-fallback path: K passed explicitly.
- `_trf_gram` — a vendored copy of scipy 1.18.1's `trf_bounds` where
  the Jacobian callback returns (G, g) instead of a matrix; the
  per-step SVD of the augmented Jacobian is replaced by ONE
  np.linalg.eigh(G_aug) (dsyevd); directions with s < gram_cut*s[0]
  truncated; termination, trust-radius updates and nfev accounting
  follow scipy exactly, so iteration counts stay comparable. Uses
  scipy PRIVATE API (scipy.optimize._lsq.common) behind a guarded
  import (_TRF_COMMON_OK) — pin scipy at the migration target or
  keep 'gram' disabled there.
- `_gram_eval_quadratic` / `_gram_build_quadratic_1d` /
  `_gram_select_step` — scipy's quadratic helpers in Gram form
  (select_step's reflection / anti-gradient machinery included).
- `_varproj_jac(..., with_k=False)` — the gram branch skips the K
  triangular solve on the QR path; the dense/operator branches are
  untouched and bit-identical.
- DIAGNOSTICS: module-level `AI_fit.gram_diag_hook` — when set,
  `_trf_gram` calls it at EVERY outer iteration with the live
  spectrum (lam, V, s, keep, n_trunc, cut, G, g, d, diag_h, x,
  cost) plus `context` = the fit_poles factor cache
  (context["core"] = Res/X/Q/R1, context["jac"] = Z/K/S), so hooks
  can rebuild the dense J and cross-check eigh against svd. Leave
  None in production; an earlier harness that monkey-patched module
  internals was REJECTED on principle — use this hook.
- Permanent regression guards:
  tests/AI_fit_test_synthetic.py::check_gram_solver (identities on
  both factor paths + a fit_poles exact-vs-gram trajectory gate).
  RERUN after any scipy upgrade.

Validation (validation/10_gram_solver/): G/g identities vs the dense
J 4e-16 .. 5e-15 (QR + forced SVD fallback; synthetic 3 shapes +
real RSSE early/late iterates); eig(G) == sv(J)^2 to 1e-14;
trust-region step vs scipy's exact SVD path worst 2.5e-14 (Delta
0.05-5x GN, active bounds, both factor paths); select_step 4e-19;
RSSE lead1 trajectory identical to exact (nfev 194=194, cost to 8
digits, poles 1e-10); synthetic gate equal minimum to 4.8e-14
relative cost. Speeds (production fit_boosted, medians of 3): RSSE
exact 27.3 / lsmr 18.6 / gram 9.2 s; Ag3Au3 lead1 exact 342.9 vs
gram 45.6 s (7.6x) — and after the refit_scales + PSD-fusion
optimization rounds the Ag3Au3 production test runs 6 fits in
204 s (was 583 s; see 4.11).

gram_cut evidence (cut_deep_dive.py, gram_diag_hook on the FULL
production fit, 7.5k outer iterations/run):

- Ag3Au3 is the only system where the cut matters; its weak
  directions sit BETWEEN 1e-7 and 1e-9 relative singular value.
- cut=1e-7 truncates only 10 directions across a whole trajectory
  (2 at the M=24 stage, 8 at M=63) — but they are REAL: at
  weak-spectrum iterations eigh and svd(J_aug) agree to 1e-6..1e-8
  with eigvec overlap 1.00. Truncating them degraded the fit
  (rms 0.0276, padded min eig -1.6e5).
- Below ~1e-8 s0 (the eps*lam_max eigh noise floor) eigh-vs-svd
  disagreements appear (rel diff up to 1.53, overlap 7.7e-5), but
  the directions kept there proved harmless: cut 1e-9 and 1e-11
  give bit-identical fits.
- The spread round (4.13) closed the question: cut 1e-9 == cut 0 in
  all 56 runs; 1e-7 actively harmful. DEFAULT 1e-9 CONFIRMED.
- The contingency design (hybrid gram + exact QR+SVD fallback when
  lam_min/lam_max < ~1e-12 on an iteration) is NOT NEEDED: the
  spread showed basin chaos hits every solver equally; the fallback
  would not reduce it.

Decisions in force (2026-10-09):

- DEFAULT tr_solver 'lsmr' (user-set); 'gram' opt-in:
  `fit(..., tr_solver="gram", gram_cut=1e-9)`.
- gram_cut default 1e-9 — confirmed by the spread (bit-identical to
  cut=0 everywhere; 1e-7 harmful).
- Ag3Au3 / Ag3Au3_gap 'exact' pin (apply_method.py per-system key):
  unsupported by the distributions — user decision pending
  (section 8). Production recipe for basin-sensitive systems:
  tr_solver='gram', 2-3 jittered starts
  (validation/11_gram_basin_spread/spread_common.make_starts), joint
  health gate (padded min eig AND T-level error), keep the passing
  fit (validation/11_gram_basin_spread/ag3au3_gram_production.py).
- OMP_NUM_THREADS=1 (re-confirmed at n=2: ~11% slower AND every
  basin shifts — see the 4.11 threading note).

### 4.15 The sweep framework (`AI_fit_sweep.py`, was sweep_framework.py) — tested on all nine systems

The multi-start / health-gate / caching / T-evaluation machinery of
the 2026-10-09/10 rounds, formalized at the repo root and
END-TO-END TESTED on every Data2 system with pickles: nogap_Ag5,
spinpol_ag5, agnr, Fig2Ag5 (the four-system round), zgnr, zgnr_spin,
Ag3Au3, Ag3Au3_gap, VerySmallRSSE (the remaining-systems round).
User guide with a usage snippet: GUIDE_3py_integration.md section
9. API:

- `GammaSweep(lf, E, F, cache_dir, base_kw, n_res, n_starts,
  lead_label, ...)` — multi-start (deterministic control + 1e-6 eV
  jittered starts), health-gated pick per config (best rms among
  healthy starts), per-fit npz caches (COMPRESSED; reruns resume),
  per-config init-aware start generation (configs touching
  init/n_init/e_margin get their own starts). `lead_label` makes
  the (lead, ik) slices of MULTI-K systems cache-disjoint - run one
  sweep per slice, in parallel background processes.
- `evaluate_pair(lf, R, pair, E_ref, T_ref, dos_ref, teta, corr,
  twin, bands, plot, E_fit, F_fit)` — inject + renormalize + banded
  T/DOS vs the sampled-SE reference + the 3-panel plot (T, DOS,
  fitted function with sharp-feature inset). MUTATES R - pass a
  freshly loaded copy per evaluation.
- `best_pair_inherit(R, name)` — symmetric-system pole inheritance
  via fit_fixed_poles (seconds; the own-fit alternative is a second
  GammaSweep - let the T level decide).
- `equalize_pole_counts` + `inject_fits(equal_nl="all")` for the
  common pole count multi-k injection requires.

WORKFLOW per system: snapshot the pristine reference ->
(optional) gradient refinement (derivative_only, do_remove=False;
see 6.3 nogap lesson) -> GammaSweep on the refined grid -> evaluate
per config at the MATCHED eta -> winner tail. Cost: ~10-25 min per
system with slices in parallel.

CROSS-SYSTEM FINDINGS (matched-eta T rms, same reference per
system):

| system | stored | framework winner | T rms | winning knobs |
|---|---|---|---|---|
| nogap_Ag5 | 0.068 | tgt1e4 + lead1 inherited | 0.0298 | tgt 1e-4, w 0.25 |
| spinpol_ag5 | 0.0397 (mismatched eta) | m1_w03, 6 starts | 0.0082 | mult 1, w 0.3 |
| agnr | 0.153 | tgt1e4 | 0.0544 | tgt 1e-4 |
| Fig2Ag5 | 0.0247 (mismatched eta) | tgt1e4 | 0.0053 | tgt 1e-4 |

- target_rms=1e-4 is the biggest single lever, winning on 3 of 4
  systems (NOT on spinpol): it drives the ADAPTIVE loop to a better
  pole set and T follows. Targets <= 1e-4 are equivalent (the
  dispatcher's absolute thresholds dominate).
- eta_floor_mult: never above 1 on these systems; 1.5 is a DEAD
  ZONE (rms 0.18-0.21 on all four); 2+ collapses sharp features.
- w_max: system-specific sharp optimum (nogap 0.25, spinpol 0.3,
  agnr 0.3, Fig2Ag5 0.3-0.5) - sweep it, do not guess.
- TETA MATCHING: two of the four stored comparisons (spinpol,
  Fig2Ag5) used teta=1e-2 against 2e-2 samplings - the mismatch
  inflates T rms ~5x and even changes the RANKING (Fig2Ag5: w03
  looks best under the mismatch, tgt1e4 at matched eta). Always
  compare at the contour eta; the stored numbers for both systems
  are superseded (6.3/6.4 updated).
- The multi-start health gate is mandatory everywhere: basin
  collapses to rms 0.18-0.21 occur on every system, and winning
  basins often appear only at starts s2-s5.
- Tight-target winners carry mild PSD violations (padded min eig
  -0.06 .. -0.32, within the gate) and huge cancelling residues
  (the Zandpack tail's +-1e5..1e8 eigenvalue print; complex-plane
  limsup stays tame) - watch the health record, do not judge by rms.
- Robustness fixes made during the tests: PaddedFit.__getattr__
  unpickle RECURSION bug (guarded; winner_fits.pkl round-trips
  now), per-config init-aware starts (init mode is a NO-OP under
  psd=True - the dispatch always uses the NNLS dictionary),
  COMPRESSED caches (np.savez_compressed; existing 283 caches
  recompressed 238 MB -> 33 MB), lead_label for multi-k slices.
  Review round (2026-10-10, Zandpack-side code review): ENVIRONMENT
  INVARIANTS NOW ENFORCED IN CODE (AI_fit_base warns at import when
  OMP_NUM_THREADS != 1; warns when gram runs on an unverified scipy
  even if the private import succeeds; requirements.txt added with
  the scipy==1.18.1 pin), evaluate_pair's RX=R alias removed (R
  mutated in place, documented), best_pair_inherit REQUIRES name=
  when the sweep holds multiple configs (no implicit cross-config
  rms pick) and raises on empty results, AI_fit_adaptivesampling
  docstring states the pickle-portability scope explicitly
  (package-era lorentz_fit caches are NOT portable by design).

**LOWPOLE round (2026-10-09, agnr + Fig2Ag5)**: same systems,
same grids, constraint = max 50 TOTAL poles per electrode
(enforced as max_poles + n_res <= 50; every fit verified <= 50;
previous-run data kept - separate sweep_*_lowpole caches and
*_lowpole artifacts). Winners: agnr lp_42p8_t4 T rms 0.0702 (29%
worse than the unconstrained 88-pole 0.0544; the BOOST split wins
there), Fig2Ag5 lp_50p0_t4 T rms 0.00533 (essentially FREE - the
unconstrained 81-pole winner was 0.00527; all-adaptive wins there).
Under the pole constraint: the tight target (1e-4) is again the
decisive knob; the adaptive/boost budget split is SYSTEM-DEPENDENT
(sweep both 42+8 and 50+0). TWO traps found and handled:
(a) fit_boosted(n_res=0) does NOT skip the boost - it pops
max_poles from the residual kwargs and runs an UNCAPPED adaptive
fit of the residual (the first 50+0 attempt silently produced 52+
poles). The framework's n_res=None now means the PLAIN adaptive
fit; the invalid caches were purged and re-run. (b) the gamma-level
rms column mixes metrics across n_res groups (fit -> FitResult.rms
= quad on scaled data; fit_boosted -> CombinedFit.rms = frobenius
relative) - rank ACROSS configs only at the T level
(evaluate_pair's frobenius_rms_error), never by the framework's
per-config rms. NOTE: the lowpole tails ran tofile('src'), which
overwrote src/ with the lowpole injection; the unconstrained
injections are preserved in tmp_fit_sweep.gz per system (rerun
inject + tofile to restore).

**LOWPOLE round 2 (nogap_Ag5 + spinpol_ag5, 2026-10-09 night)**:
same 50-total-pole constraint on the remaining two framework
systems (nogap on the finer 1105-pt grid; spinpol multi-k on the
original 351-pt grid; all fits verified <= 50, equalized across
k-points; previous data untouched). Winners: nogap lp_50p0 T rms
0.1067 (+258% vs the unconstrained 0.0298 — the van Hove streak
NEEDS its ~96 poles; PSD of the 50-pole winner clean, min eig
-0.0), spinpol lp_42p8_t4_w05 T rms **0.00790 — BETTER than the
unconstrained 72-pole winner (0.00824)**. NEW INSIGHT: the w_max
optimum SHIFTS with the pole budget (spinpol: w03 optimal at 72
poles, w05 at 50 — fewer poles favor BROADER poles; sweep w_max
AGAIN whenever the budget changes). Full four-system lowpole
matrix (T rms at <=50 total poles per electrode vs unconstrained):

| system | lowpole winner | T rms | unconstrained | cost of 50 poles |
|---|---|---|---|---|
| agnr | lp_42p8_t4 (boost split) | 0.0702 | 0.0544 (88) | +29% |
| Fig2Ag5 | lp_50p0_t4 (all adaptive) | 0.00533 | 0.00527 (81) | free |
| spinpol_ag5 | lp_42p8_t4_w05 | 0.00790 | 0.00824 (72) | BETTER |
| nogap_Ag5 | lp_50p0 | 0.1067 | 0.0298 (96) | +258% |

Budget-split rule under constraint: sweep BOTH 42+8 and 50+0 (agnr
and spinpol want the boost; Fig2Ag5 and nogap want all-adaptive);
34+16 always loses. Artifacts per system: cmp_fits_lowpole.npz,
method_summary_lowpole.json, tmp_fit_lowpole.gz,
sweep_*_lowpole/ (results_lowpole.json + top-3 plots).

**Remaining-systems round (2026-10-10: zgnr, zgnr_spin, Ag3Au3,
Ag3Au3_gap, VerySmallRSSE)**: the same pipeline applied to every
remaining Data2 system with pickles (pp_psf is pseudopotentials
only; AuBreakJunction has none). Full record: Data2/
README_remaining_round.md + per-system sweep_* dirs. Results
(matched-eta T rms): zgnr m1_forc_w03 0.0891 (refined grid),
zgnr_spin m5_forc 0.1309 (matched eta; stored-teta 0.376), Ag3Au3
m1_w03 **0.006228**, Ag3Au3_gap m5_exact 0.003487 (m1_w03 0.003636
tied), VerySmallRSSE m1_w03 0.02007. FOUR durable findings:
(1) **Ag3Au3/Ag3Au3_gap EXACT-PIN RESOLVED** (the HANDOFF 8 open
item): on the gradient-refined grids with the 3-start health gate,
gram m1_w03 BEATS the pinned exact on Ag3Au3 (0.00623 vs 0.00699)
and is statistically tied on the gap variant (0.00364 vs 0.00349)
at ~5x less wall. Recommendation: drop the 'exact' pin in
Data2/apply_method.py CFG in favor of gram + multi-start health
gate. The 2026-10-09 gram production record (0.007719 on the
UNREFINED grid) is itself beaten by the refined-grid m1_w03.
(2) **m1_w03 (mult=1, w_max=0.3) is the strongest cross-system
default** - winner on Ag3Au3 + RSSE, top-2 on Ag3Au3_gap + zgnr.
(3) **zgnr's stored 0.070 is NOT reproducible**: same knobs on the
original grid via the framework give 0.110; the refined grid is
better (0.0891). The stored number was a favorable single-run
exact-era basin. ALSO a GATE-CALIBRATION lesson: zgnr's honest fit
rms (~0.08-0.14 frobenius) exceeds the default rms_max=0.05 health
threshold, so every zgnr pick was "best of bad lot" - loosen
rms_max per system (never min_eig_tol).
(4) **zgnr_spin is the THIRD eta-mismatched system** (2e-2 contour,
stored teta 1e-2: 0.376 vs 0.131 matched); and its gradient
refinement had to be STOPPED - the refine_loop's FINAL fit spent
>55 min in a pathological TRF basin (p=46 -> 2116 channels; same
failure mode as w30_exact s1 on nogap). The sweep ran on the
original 121-pt grid. Pathological-basin fits have no cheap
in-flight guard yet - a max wall per fit knob for refine_loop is an
open improvement item.

### 4.16 Recipe: getting high-quality fits (distilled from the four-system rounds)

Ordered checklist for a NEW system. Every line below is measured,
not opinionated; references point to the sections with the numbers.

1. SAMPLING: gradient-based refinement FIRST — refine_loop(
   derivative_only=True, do_remove=False; guide 6.4; the removal
   round is counterproductive on deliberately dense sharp-feature
   grids). A finer mesh alone does NOT improve the T level (w30 got
   worse on nogap) — refine only where features are under-resolved.
2. SOLVER: tr_solver='gram', gram_cut=1e-9. Fastest, and at the T
   level equal-or-better than lsmr (10x slower AND worse on the
   test systems) and exact (pathological basins) on all four.
3. MULTI-START + HEALTH GATE, always: 3-6 jittered starts
   (sweep_framework.GammaSweep); judge by padded min eigenvalue AND
   sample rms TOGETHER. Basin collapses to rms ~0.2 occur on every
   system; winning basins often appear only at starts s2-s5.
4. KNOBS, in order of measured leverage:
   a. target_rms=1e-4 — the biggest single lever (drives the
      adaptive loop to a better POLE SET and T follows; targets
      <= 1e-4 are equivalent). Won on 3 of 4 systems.
   b. w_max_raw — system-specific SHARP optimum (nogap 0.25, spinpol
      0.3, agnr 0.3); sweep 0.2-0.5 in 0.05 steps. It SHIFTS with
      the pole budget (broader when poles are scarce) — re-sweep
      after any budget change.
   c. eta_floor_mult=1.0 — never higher on these systems (1.5 is a
      dead zone everywhere; 2+ collapses sharp features).
   d. max_poles — give the loop room (winners used 50-150); if a
      pole BUDGET applies, see the lowpole matrix in 4.15.
   e. n_init=24; tail anchors ON; psd_force=False (True only where
      the nanoribbon-style PSD-at-all-costs policy is needed).
      INERT everywhere tested: local_floor_frac, ridge,
      prune_rel_tol, the init MODE (no-op under psd=True), and
      n_res when the budget is not binding.
5. EVALUATE at the CONTOUR eta (teta = sampling eta), on T + DOS +
   the fitted function vs sampled (evaluate_pair + its plots).
   Two of four stored suite comparisons were inflated ~5x by an eta
   mismatch, which also changed their rankings. Rank by the T level,
   never by the mixed .rms attributes (CombinedFit vs FitResult
   metrics differ).
6. SYMMETRIC systems: lead1 by POLE INHERITANCE (fit_fixed_poles;
   ridge=0 default; MATCH the source fit's psd policy) — equal or
   better than lead1's own full fit at ~1/50 cost (guide 8).
7. MULTI-K systems: one GammaSweep per (lead, ik) slice
   (lead_label), equalize_pole_counts across all slices before
   inject_fits (one common NumL is a Zandpack requirement).
8. HEALTH of the winner: read the padded min eigenvalue (gate
   threshold -1) and expect large cancelling residues on
   tight-target high-pole fits (the +-1e5..1e8 tail prints are
   common to ALL such fits; complex_plane_limsup stays tame when
   the fit is good).
9. TRAPS (all hit at least once in these rounds): fit_boosted
   n_res=0 runs an UNCAPPED residual fit (use n_res=None for no
   boost); CombinedFit.rms vs FitResult.rms metric mismatch;
   the removal round on dense grids; teta != sampling eta (THREE
   stored systems were mismatched: spinpol, Fig2Ag5, zgnr_spin);
   single-start comparisons on basin-sensitive systems; and
   PATHOLOGICAL-BASIN FITS that run for hours (zgnr_spin's
   refinement final fit, p=46; nogap's w30_exact s1) - there is no
   in-flight wall guard yet (open item); when a fit is taking
   abnormally long, stop it and take a different start.


## 5. How to run

```bash
cd FittingRoutine
# ported suites (import AI_fit; AI_FIT_MODULE env var switches the
# import target — default repo-root AI_fit_base.py, or a relocated copy):
python3 tests/AI_fit_test_synthetic.py   # core tests (ALL PASS)
python3 tests/AI_fit_test_model_form.py  # plan-exactness proof (ALL PASS)
# (tests/test_synthetic.py + test_model_form.py are the original
#  package-era suites, kept for reference; the ported ones are the
#  active gate.)

cd bench
python3 plot_examples.py               # detail figures (refits, ~2-4 h)
python3 performance_figure.py          # aggregate overview (fast, reads JSON)
python3 run_fit_bench.py <example> [--psd] --json out.json

# basin-spread round (4.13; the machine is ONE core, run sequentially):
OMP_NUM_THREADS=1 python3 validation/11_gram_basin_spread/rsse_spread.py      # ~8 min
OMP_NUM_THREADS=1 python3 validation/11_gram_basin_spread/ag3au3_spread.py    # ~2 h
OMP_NUM_THREADS=1 python3 validation/11_gram_basin_spread/ag3au3_gram_production.py  # ~4 min
OMP_NUM_THREADS=1 python3 validation/11_gram_basin_spread/analyze.py --plot

# gram solver validation suite (4.14):
OMP_NUM_THREADS=1 python3 validation/10_gram_solver/identities.py
OMP_NUM_THREADS=1 python3 validation/10_gram_solver/step_agreement.py
OMP_NUM_THREADS=1 python3 validation/10_gram_solver/trajectory.py     # ~5 min
OMP_NUM_THREADS=1 python3 validation/10_gram_solver/ag3au3_gate.py     # ~7 min
OMP_NUM_THREADS=1 python3 validation/10_gram_solver/cut_deep_dive.py    # ~5 min (walls incl. svd cross-check - not for timing)

# THE MODERN WORKFLOW (2026-10-09/10 rounds; 4.15/4.16 + GUIDE):
#   GUIDE_3py_integration.md       - port a 3.py: template + recipes
#   AI_fit_sweep.py                - GammaSweep + evaluate_pair (usage:
#                                   GUIDE section 9); per-system
#                                   pipeline scripts in each Data2
#                                   system folder (zgnr_pipeline.py,
#                                   ag3au3_pipeline.py, rsse_pipeline.py,
#                                   zs_slice_*.py, ...) demonstrate the
#                                   full refine -> sweep -> eval -> tail
#   lowpole runs: the same framework with max_poles + n_res <= 50
#   (n_res=None = plain adaptive fit; NEVER n_res=0 - uncapped
#   residual fit, 4.15)
```

Traps (standing): load pickles BEFORE any repo-root sys.path insert
(the shadow rule, 3/4.13); the gram-solver suite hooks must use the
documented gram_diag_hook, never monkey-patching; keep
OMP_NUM_THREADS=1 (4.11).

Sampling refinement (Data2 examples — see section 4.8):

```bash
cd Data2/zgnr        && python3 refine.py    # needs tmp.gz (2.py run once)
cd Data2/zgnr_spin   && python3 sample.py    # no tmp.gz yet: first sampling
                     && python3 refine.py
cd Data2/VerySmallRSSE/refined && python3 refine.py   # rebuilds per round
```

Applying the current method + old-vs-new comparison to any remaining
Data2 system (section 6.3):

```bash
cd Data2/<system> && python3 ../apply_method.py <system>
# e.g. agnr, spinpol_ag5, Fig2Ag5, Ag3Au3, Ag3Au3_gap, AuBreakJunction
```

Performance notes: keep `OMP_NUM_THREADS=1` (4.11); fits are silent by
default — pass `verbose=True` in the fit kwargs to watch every stage
(the "looks hung" experience is a fit computing, not a hang —
although a fit taking >30 min IS usually a pathological basin, 4.16
item 9: stop it and take a different start). Single fits are fast
post-4.11/4.14; the SWEEP rounds cost more wall by design
(multi-start x configs x slices, ~10-90 min per system in parallel
slice processes).

Each writes `refined/` figures + `summary.json` (zgnr*, and directly in
the folder for VerySmallRSSE) plus a refreshed pickle
(`tmp_refined.gz`) whose stored sampling IS the refined grid.

Figures land in `bench/results_lorentz_fit/figures/` (12 per-example
figures + `performance_overview.png` + `summary.json`).

## 6. Benchmark status (60-pole cap)

IMPORTANT (block-storage fix, section 4.7.1): the numbers below are the
CURRENT benchmark — a single run on the CORRECTED data assembly (the
stored Gammas' lower-triangle off-diagonal blocks are no longer halved;
true Γ is fitted). The pre-fix tables are preserved as
`summary_halved_era_merged_best.json` (merged best) and
`summary_localfloor_run.json` in the same figures/ directory; the
corrected run is `summary_corrected_blocks.json`. Single-run BLAS
basin variance (caveat 1) still applies — e.g. fig2ag5 L1 and spinpol
L1 landed in poor basins this run.

| example / lead | rms | poles | λ_min total, in window |
|---|---|---|---|
| SmallRSSE L0 / L1 | 0.0050 / 0.0006 | 60 / 19 | +0.020 / ~0 |
| Ag3Au3 L0 / L1 | 0.0019 / 0.0031 | 60 / 57 | −0.003 / −0.009 |
| Ag3Au3_gap L0 / L1 | 0.0018 / 0.0052 | 60 / 58 | −0.004 / −0.028 |
| Fig2Ag5 L0 / L1 | 0.0062 / 0.0386 | 37 / 44 | −0.030 / −0.107 |
| spinpol L0 / L1 | 0.0029 / 0.0462 | 54 / 31 | −0.003 / −0.185 |
| AuBreakJunction L0 / L1 | 0.0238 / 0.0168 | 60 / 60 | −1.9 / −2.7 |

Notes on the correction's effect: leads whose off-diagonal blocks were
previously halved now fit the physically doubled coupling — smallrsse
L0 got genuinely harder (0.0011 → 0.0050 at the 60-pole cap) while
ag3au3 / ag3au3_gap / spinpol L0 / aubj came out equal or better.
fig2ag5 L1 and spinpol L1 show large single-run basin variance; their
pre-fix merged-best values were 0.0030 / 0.0051 (also on halved data —
treat any cross-era comparison as approximate).

Off-window λ_min (±30% padded grid) stays bounded at percent scale of
the data — compare with −10³ … −2.7·10³ before the defense stack.
Runtimes: 30 s (p=6) to ~35 min (p≈60, two-stage floor); aubj p=85/90
uses channel filtering (1e-4). Improvement over attempt-1's best raw
fit remains 3–50× depending on example. The aggregate figure is
`bench/results_lorentz_fit/figures/performance_overview.png`;
per-example figures and `summary.json` (corrected run) live in the
same directory.

### 6.1 Data2 sampling-refinement results

"sample rms" = fit vs its own refined sampling; "ref-grid rms" = fit
evaluated on the ORIGINAL example grid (honest out-of-sample check).
Figures: `Data2/<example>/refined/` (+ `summary.json` there).

zgnr (TBtrans path; coarse 63-pt start → 3 rounds → removal →
100 points vs the original 251; run 2080 s):

| lead | pts (orig) | poles | sample rms | ref-grid rms | 251-pt ref fit |
|---|---|---|---|---|---|
| 0 | 100 (251) | 47 | 0.127 | 0.156 | 43 poles, 0.092 |
| 1 | 100 (251) | 37 | 0.154 | 0.192 | 52 poles, 0.108 |

Off-window excursion: 1.4–1.7% of data scale (no spikes). Between-sample
λ_min: the sharp peak at E≈−0.056 (Tr jumps 133→23 over 0.056) is
resolved BETTER by the refined grid (λ_min −0.65 within |E|<0.5 vs −6.9
for the full uniform grid); the worst refined-grid dip (−69 at E≈−2, in a
coarse gap) is the Hermitian-but-indefinite-residue freedom between
samples. `psd=True` makes the model PSD everywhere by construction:
lead1 fully PSD at 2× rms cost; lead0's PSD projection was rejected by
the accuracy guard (unconstrained model kept). Physical resolution of
between-sample indefiniteness = denser sampling there (lower
`insert_tol`), or accept the PSD-fit accuracy cost.

zgnr_spin (mimic_tbtrans_spin runner, 2 spins; coarse 61-pt start →
3 rounds → removal → 91 points vs the original 121; run 9079 s):

| lead / spin | poles | sample rms | ref-grid rms |
|---|---|---|---|
| 0 UP | 34 | 0.036 | 1.30 |
| 0 DN | 53 | 0.124 | 0.169 |
| 1 UP | 51 | 0.0030 | 0.0162 |
| 1 DN | 47 | 0.0034 | 0.0120 |

lead0 UP out-of-sample (1.30) is the weakest of the set — the 91-pt grid
under-resolves its structure between samples; lower insert_tol (e.g.
2e-3) and/or more rounds is the remedy (knob documented in 4.8).

(NOTE: the VerySmallRSSE fit numbers in this subsection predate the
block-storage fix (4.7.1) and the psd_force default — lead0's
off-diagonal blocks were halved then. The authoritative VerySmallRSSE
quality numbers are the transmission comparison in 6.2. zgnr and
zgnr_spin have complete block sets and are unaffected.)

VerySmallRSSE (rebuild-per-round via parameterized make_device.py;
151 → 161 → 167 → 171 → removal 75 → 96 points; run 1497 s ≈ 2.5 min
per rebuild, dominated by Real_space_SE):

| lead | pts (orig) | poles | sample rms | ref-grid rms | orig-grid fit |
|---|---|---|---|---|---|
| 0 | 96 (151) | 44 | 0.00103 | 0.0147 | 0.00113 |
| 1 | 96 (151) | 19 | 0.0023 | 0.0023 | 0.0016 |

lead0's refined 96-point fit is slightly BETTER on its own samples than
the original 151-point fit; out-of-sample fidelity tracks insert_tol
(5e-3 → ~1.5e-2 ref-grid error). Half the points at equal sampled
fidelity — the practical win the refinement loop is for.

### 6.2 DOS / transmission comparison (old pipeline vs refined fit)

`compare_dos_t.py` in each example folder runs BOTH the old 3.py /
make_fit.py pipeline (on the original sampling) and the new refined
psd-forced fit through the identical Zandpack post-processing —
find_correction -> Renormalise_H ->
Inspect_transmission_from_hilbert_transform /
Inspect_dos_from_hilbert_transform. References: the SAMPLED
self-energies in the same sub-orbital space
(`sampled_transmission_dos`, Fermi-pole points excluded via
sampling_idx) for zgnr/zgnr_spin; TBtrans for VerySmallRSSE (no
sub_orbital truncation there). The nanoribbon comparisons are
restricted to E >= -3: the sub_orbital selection removed the
orbitals that carry the transmission below -3, so any T in the
truncated space is meaningless there (fit and reference alike).
Figures: `Data2/*/refined/dos_t_comparison.png`.

Final numbers (width policy eta_floor_mult=5.0, psd_force + residual
boost n_res=8 (4.9); zgnr/zgnr_spin vs the sampled-SE reference over
E >= -3, VerySmallRSSE vs TBtrans over +-5). The zgnr NEW column is
the WIDE-WINDOW fit (sampling extended to [-12,12], see 4.9); zgnr
uses w_max_raw=0.35, zgnr_spin needs 0.5 (0.35 REGRESSES it: 0.266
vs 0.216 - broad spin-polarized lead features), VerySmallRSSE 0.35:

| system | old pipeline | new fit | notes |
|---|---|---|---|
| zgnr | 0.189 | **0.070** -> **0.0891** (2026-10-10 round; the stored 0.070 NOT reproducible - lucky exact-era basin; same knobs on the original grid give 0.110, the refined grid 0.0891) | sharp feature: new 0.813, old 0.811, ref 0.812; wide-window fit |
| zgnr_spin | 0.454 | **0.216** -> **0.1309** at MATCHED eta 2e-2 (2026-10-10; stored-teta column 0.376 - third eta-mismatched system) | better in every band; 32-38 poles [24-30+8] |
| VerySmallRSSE | 0.090 | **0.0079** -> **0.0201** on the 151-pt base grid, m1_w03 (2026-10-10; the stored 0.0079 used the edge-DENSIFIED 189-pt grid - not comparable) | psd_force=False (notch); T(0) 0.0212 vs 0.0237 |

All three refined fits BEAT the old pipelines' transmission accuracy
while being PSD everywhere (Gamma >= 0 also outside the window) and
using 40-75% of the sampling points.

History of this section: the first comparisons (TBtrans reference,
full window) were dominated by (a) the sub-orbital truncation below
E=-3 on the nanoribbons, and (b) two fit pathologies later fixed -
the lower-triangle block storage (4.7.1) and the width policy (4.9).
VerySmallRSSE stages: 0.876 (pre block-fix) -> 0.108 (pre width
policy) -> 0.049 (pre boost) -> 0.045 -> 0.067 (notch fix, edges
still ringing) -> 0.0079 (edge densification, final). zgnr stages:
0.170 (pre boost) -> 0.147 -> 0.130 (w_max 0.35) -> 0.070 (window
extension, final); sharp-feature fidelity: 0.474 -> 0.598 -> 0.810
-> 0.813 (reference 0.812).

### 6.3 The current method applied to every remaining Data2 system

Driver: `Data2/apply_method.py <system>` (run from inside the system
folder). Per system it replicates that folder's own 3.py/Fit.py old
pipeline verbatim on a pristine pickle, fits the same sampling with
the current method (PSD + per-system width policy + residual boost,
setLval injection), and compares T/DOS against the sampled-SE
reference. Figures + numbers: `Data2/<sys>/method_comparison.png`,
`method_summary.json`.

CAVEAT (2026-10-09/10 rounds): the apply_method comparisons used a
per-system `teta` that is MISMATCHED against the contour eta for
THREE systems (spinpol_ag5, Fig2Ag5, zgnr_spin: teta 1e-2 vs 2e-2
sampling) - inflating those stored numbers ~5x and changing their
rankings. The superseding sweep-round numbers in the table (and in
the per-system sweep_* dirs) are all at the MATCHED eta. The modern
way to produce these numbers is the sweep framework (4.15), not
apply_method's single-start path.

| system | old 3.py | lorentz_fit | policy notes |
|---|---|---|---|
| agnr | 0.245 | **0.153** -> **0.0544** (2026-10-09 sweep round) | mult=1 (armchair features narrower than zgnr's: mult=5 gave relrms 0.52, mult=1 gives 0.0145); winner tgt1e4 on the gradient-refined 347-pt grid |
| spinpol_ag5 | 0.103 | **0.0397** | mult=2, psd_force=False — SUPERSEDED by the 2026-10-09 sweep round, see below |
| Fig2Ag5 | 0.0254 | **0.0247** (mismatched teta) -> **0.0053** at matched eta (2026-10-09 sweep round) | winner tgt1e4; stored teta=1e-2 vs 2e-2 sampling inflated the old comparison ~5x and changed the ranking |
| Ag3Au3 | 0.0383 | **0.0100** -> **0.00623** (2026-10-10: refined grid + gram m1_w03 + 3-start gate; BEATS the exact pin 0.00699 - pin resolved) | mult=1, w_max 0.3, psd_force=False |
| Ag3Au3_gap | 0.00785 | **0.00346** -> **0.00364** gram m1_w03 (exact 0.00349 - statistically tied; 2026-10-10) | pin droppable here too |
| AuBreakJunction | 0.084 | **0.0483** | DOS + oscillation metric also better; sampling still under-resolved (caveat 4) |
| nogap_Ag5 | 0.113 | **0.068** | mult=1, w_max=0.3 (sweep: 0.068 vs 0.080 at w_max 0.5), mx=62+8 (70 total); eta-matched comparison — see note below |

**Full lsmr-default sweep (2026-10-09, fresh fits, all systems)**: with
`tr_solver='lsmr'` as the global default the fresh-fit T rms vs the
sampled-SE reference came out: agnr 0.153, Fig2Ag5 0.026,
spinpol_ag5 0.045, AuBreakJunction 0.044 (BETTER than exact-era
0.048), nogap_Ag5 0.063 (better than exact-era 0.068), zgnr 0.073,
zgnr_spin 0.223, VerySmallRSSE 0.0087 — equal-or-better on 8/10.
REGRESSION — the Ag3Au3 FAMILY: Ag3Au3 0.065 (exact-era 0.0100;
minEig -1.1e3) and Ag3Au3_gap 0.019 (exact-era 0.0035; minEig -65):
lsmr lands in a non-PSD basin that these systems' psd_force=False
guards accept. apply_method now PINS `tr_solver='exact'` for both
(the exact path is bit-for-bit unchanged, so their stored numbers
remain valid). LESSON: for psd_force=False systems, check the padded
minEig when trying lsmr. Fit walls this sweep (old pipeline
excluded, approx): agnr 73s, Fig2Ag5 530s, spinpol 270s, AuBJ 1500s,
nogap 800s, Ag3Au3 680s (lsmr; exact-era ~540+ per lead), Ag3Au3_gap
1080s, zgnr 200s, zgnr_spin 150s, RSSE 30s — the entire 10-system
sweep ran ~5.3 h single-core. Caches: the exact-era fits are
preserved as cmp_fits_exactera.npz / *_exactera.npz next to the
fresh cmp_fits.npz in every system folder.

The width/mult policy is genuinely per-system: the metals take
mult=5/w_max=0.5 with psd_force=False, the armchair nanoribbon needs
mult=1, the eta=0.02 systems mult 1-2. With the performance work of
4.11 the full six-system sweep runs in well under an hour on one core
(the pre-optimization Ag3Au3 pair alone took over an hour).

### 6.4 Tuned knobs per system (what was actually used)

Every number below was chosen by measurement (sweeps in the system
folders: `Data2/zgnr/diag_sweep*.py`, `Data2/agnr/policy_sweep*.py`,
`Data2/VerySmallRSSE/refined/notch_*.py`, `improve_*.py`).
The width policy is genuinely SYSTEM-DEPENDENT — there is no
universal eta_floor_mult/w_max.

| system | eta_raw | mult | w_max | psd_force | target / max poles | sampling decision |
|---|---|---|---|---|---|---|
| zgnr | 1e-2 | 5.0 (historical) / **1.0** (2026-10-10 winner is m1_forc_w03) | **0.3** (2026-10-10) | True | 2e-3 / 100 | refined 347-pt grid (gradient round); stored 0.070 NOT reproducible - see 4.15 remaining-round entry |
| zgnr_spin | 2e-2 | **5.0** (2026-10-10 winner m5_forc; m1 close) | 0.5 | True | 2e-3 / 100 | original 121-pt grid (refinement final fit hit a pathological basin, stopped); MATCHED teta 2e-2 - stored comparisons used 1e-2 |
| VerySmallRSSE | 1e-2 | **1.0** (2026-10-10 winner m1_w03; 3.0 historical) | **0.3** (2026-10-10) | **False** (PSD-forcing fills the E=0 notch) | 2e-3 / 60 | 151-pt base grid (no in-place refinement possible - RSSE rebuild pattern, guide 6.3) |
| agnr | 1e-2 | **1.0** (5 catastrophic: relrms 0.52 -> 0.0145 at 1; 1.5 is a dead zone everywhere) | 0.5 | False | 1e-3 / 150 | gradient-refined 347-pt grid (2026-10-09); winner tgt1e4 T 0.0544 |
| spinpol_ag5 | 2e-2 | **1.0** (2.0 was tuned under the MISMATCHED-eta metric) | **0.3** | False | 2e-3 / 100 | original 351-pt grid; 2026-10-09 sweep: T rms 0.0082 at matched eta |
| Fig2Ag5 | 2e-2 | **1.0** | **0.3-0.5** (sweep-insensitive) | False | 1e-4 target wins / 100 | gradient-refined 553-pt grid; T 0.0053 at MATCHED teta 2e-2 (stored numbers used 1e-2) |
| Ag3Au3 / Ag3Au3_gap | 1e-2 | **1.0** (2026-10-10 winner m1_w03; the historical 5.0 + exact pin is RESOLVED - see section 8) | **0.3** | False | 2e-3 / 100 | gradient-refined 639-pt grid (2026-10-10); gram + 3-start health gate |
| AuBreakJunction | 1.5e-2 | 2.0 | 0.5 | False | 2e-3 / 100 | 251-pt grid (still under-resolved — caveat 4) |
| nogap_Ag5 | 2e-2 | 1.0 | **0.3** (0.5 gave T rms 0.080 vs 0.068) | False | 2e-3 / 62+8=70 | uniform 1001-pt grid [-8,10] (tmp.gz is the "plot" branch of 2.py) |

**nogap_Ag5 specifics** (added 2026-10-08): Ag chain with a van Hove
spike at E=3.30 (Tr Gamma up to 308, width ~0.05) and a sharp band
edge at E=-3 (plateau Tr~7.7 falls to ~0.09). Comparison-eta lesson:
the reference is sampled at eta=0.02; evaluating fits at eta=1e-2
manufactured a positive bias everywhere (T rms [-5,-3) was 0.355 at
eta 1e-2 vs 0.031 at matched eta 0.02) — for systems with van Hove
streaks sharper than the sampling eta, `teta` MUST equal the contour
eta. At matched eta the new fit beats the old in the spiked region
[-5,-3): 0.031 vs 0.057, and overall 0.080 vs 0.113; the old fit
remains slightly better in the cliff bands [-3,-2), [4,6). Pole-cap
sweep: 150/78/75-pole fits gave 0.201/0.202/0.199 at eta 1e-2
(indistinguishable); 70-pole cap (mx=62 + n_res=8) is the default.
n_res=12/mx=58 variant was worse (0.093). Caches:
`cmp_fits.npz` (final), `cmp_fits_{mx150,mx70,nres12}.npz` backups.
POST-SWEEP UPDATE: parameter sweep (sweep_fit.py Gamma-level +
sweep_stage2.py T-level; fits in sweep_fits/) adopted w_max=0.3:
overall T rms 0.0684 (0.5: 0.0803; 0.4: 0.0829 — non-monotonic basin
noise; old pipeline 0.1126). mult=2/3 CATASTROPHIC (adaptive loop
collapses at 15/13 poles — floor exceeds the sharp features);
psd_force=True identical to False (fit already PSD-compatible);
mx=40 worse, mx=80 marginal (over the 70 cap). w_max=0.3 tradeoff:
plateau band [-5,-3) 0.042 vs 0.031 at 0.5 (both still beat old
0.057). Final caches: cmp_fits.npz (w03), cmp_fits_w05.npz.

**nogap_Ag5 full-guide application (2026-10-09, post-2.py-rerun)**:
after the Device.SiP path fix the fresh 2.py produced the NON-plot
457-pt contour (0.009 spacing in [2,4]) — old stored numbers reference
the retired 1001-pt grid and are NOT comparable. Applied
GUIDE_3py_integration.md end-to-end (see Data2/nogap_Ag5/
README_gram_refined.md for the full record): (1) REFINEMENT — the
leave-one-out removal round is COUNTERPRODUCTIVE on this system (it
flagged the deliberately dense van Hove streak as redundant at BOTH
tol 1e-2 and 1e-3, orig-grid rms 0.371/0.103); the working recipe is
gradient-based insertion (derivative_only=True) with do_remove=False:
457 -> 553 pts, all 96 insertions inside the streak [3.20,3.48],
orig-grid rms 0.036/0.026 (refine_deriv.py, 91 s). (2) GRAM SWEEP —
tr_solver='gram' @1e-9, health-gated multi-start (3-6 starts): mult
2/5 catastrophic (confirmed on the new grid), w_max 0.3 > 0.5, lead1
wanted >62 poles (92-96 at mx=90), best basins found only at start s3
(basin spread is large here too). (3) WINNER — a MIXED per-lead pair
(legal: fits are independent per lead): l0 = m1_w03_mx90/s3 (92
poles) + l1 = m1_w02_mx90/c0 (96 poles), T rms 0.0492, DOS rms 1.556,
fit rms 0.0106/0.0078, padded min eig -7e-16. Best single-config
alternative: w03_mx90 both leads, T 0.0557. Artifacts:
cmp_fits_gram_refined.npz, method_summary_gram_refined.json,
T_gram_refined.png, tmp_fit_final.gz, sweep_gram/ (fit caches +
results{,2,3}.json). Total wall ~25 min (refine + 3 sweep rounds +
final). New guide lessons recorded in GUIDE_3py_integration.md 6.4 +
pitfalls (removal counterproductive on dense sharp-feature grids;
per-lead config mixing in sweeps).

**Fixed-poles re-solve finding (2026-10-09, nogap_Ag5; CORRECTED same
day)**: NEW PUBLIC API `AI_fit.fit_fixed_poles(E, F, centers, widths,
psd=True, psd_force=True, ridge=0.0, ...)` fits ONLY the residue
coefficients at FIXED poles (the final-solve + psd_refine machinery
of _fit_impl: same weights, tail anchors, PSD accuracy guard) - built
for POLE INHERITANCE in symmetric systems (fit one electrode fully,
hand the partner the fitted (e,w), solve its matrices in seconds;
nogap_Ag5's leads are structurally symmetric - lead0's 9 extra
orbitals are IDLE, max|F|=0, though the shared 18x18 blocks are NOT
identical, rel diff 1 - and inheritance fits lead1 as well as its own
full fit). Validation produced THREE durable findings:
(1) METRIC TRAP (correction of the first-reported "9x re-solve
improvement" / "12x Ag3Au3 degradation" - BOTH were artifacts):
CombinedFit.rms (what fit_boosted reports) is the RELATIVE FROBENIUS
rms, FitResult.rms is the QUADRATURE-weighted rms on scaled data;
never compare the two across objects (recheck.py round 1 did and
produced both false findings). Honest numbers (one metric,
frobenius): the re-solve at a fit's own poles is CONSISTENT - nogap
l0 0.01064 -> 0.01006 (5% better), Ag3Au3 l0 0.004976 -> 0.004812
(3% better).
(2) RIDGE TRAP: fit_fixed_poles initially defaulted ridge=0.05
(mirroring _fit_impl, where the ridge only guards a solve whose poles
are about to be RELOCATED). At FIXED poles it biases the re-solve
whenever the pole set overlaps and needs cancelling coefficients -
measured Ag3Au3 l0 (100 poles, psd=False): x24 rms degradation at
ridge 0.05 vs x0.97 at ridge 0. DEFAULT CHANGED TO ridge=0.0
(docstring documents the reasoning;
backups/AI_fit_2026-10-09_pre_fit_fixed_poles.py has the pre-API
state).
(2b) PSD-POLICY CONSISTENCY (recheck2, frobenius metric, ridge 0,
44 cases): the unconstrained re-solve is consistent everywhere
(nogap median ratio 0.91, i.e. ~10% better than the returned model;
Ag3Au3 x0.97). The remaining degradations are all psd_force COSTS,
not solve failures: forcing the PSD cone at FIXED poles on a fit
whose residues rely on inter-pole cancellation is genuinely worse
(Ag3Au3 picks: x23 / x7.3 - the adaptive fit's PSD path would ADD
poles to compensate; a fixed-poles solve cannot). Usage rule: for
pole inheritance, MATCH the source fit's psd policy - psd_force=True
only when the source fit's PSD model was applied (nogap_Ag5: yes,
inheritance+PSD works perfectly, all re-solved models PSD to
-1e-15); psd=False for cancellation-reliant fits (Ag3Au3 family).
(3) T-level invariance: the (honest, 5%) polish leaves T/DOS
unchanged (T rms 0.0492 -> 0.0500, DOS 1.556 -> 1.582): once the fit
reproduces the samples to ~1%, the transport error is limited by
model shape / window / eta / renormalization, NOT sample rms - fit
rms alone must not be the sweep tiebreaker (health gate + T level
decide). Consistency table across all cached cases:
Data2/nogap_Ag5/recheck_results.json (round 1, mixed metrics -
historical) and recheck2_results.json (corrected: frobenius metric,
ridge 0).

**nogap_Ag5 thorough round (2026-10-09, later that day)**: finer mesh
(gradient-refined 553 pts midpoint-split to 1105, tmp_finer.gz) + a
13-config coordinate sweep (w_max 0.15-0.35 in 0.05 steps, mult 1.5,
mx 110, psd_force, n_res 12, lsmr/exact/absolute-weight variants;
sweep_finer/, 52 cached fits) + pole inheritance via fit_fixed_poles
(lead1 = lead0's poles, seconds; vs lead1's own full multi-start
fit). WINNER: w_max=0.25 with INHERITED lead1 - T rms 0.0432
(previous best 0.0492 on the 553-pt grid; old 3.py 0.113 vs the old
reference), DOS rms 1.488, 96+96 poles, PSD to -1e-15, inherit solve
4 s vs ~5 min full fit. Inheritance matched or beat the own-fit at
T level in 3/5 configs (w25: 0.0432 vs 0.0749) - consistent poles
across mirror leads are physically right. Finer-mesh finding: NO
systematic T-level gain (w30 got WORSE, 0.0557 -> 0.1048, basin
shift) - consistent with finding (3): T is model-shape-limited; the
win came from the finer KNOB grid (w_max 0.25 untried before), not
the finer mesh. Solver at T level: gram > lsmr (~10x faster AND
better: w20 family 0.064 vs 0.139) and > exact (0.095; one exact fit
went pathological >45 min and was killed). m15 (mult 1.5)
catastrophic on the finer grid too. Per-variant plots (T + DOS +
Tr Gamma with van Hove inset) in sweep_finer/fig_*.png; final
artifacts cmp_fits_final.npz / method_summary_final.json /
tmp_fit_final2.gz; full record in Data2/nogap_Ag5/
README_gram_refined.md (round 4-5 section). Guide: symmetric-system
inheritance recipe = GUIDE_3py_integration.md section 8.

**nogap_Ag5 experiment rounds + sweep framework (2026-10-09
night)**: the sweep machinery is now a REUSABLE TOOL,
`AI_fit_sweep.py` at the repo root (was sweep_framework.py; GammaSweep: multi-start,
health-gated, cached, per-config init-aware start generation;
evaluate_pair: T/DOS + per-variant plots; usage in GUIDE section 9).
Dogfooded on nogap_Ag5 (experiments2/3.py) to sweep the previously
unexplored fit() options. Findings: (a) target_rms is the
highest-leverage UNTRIED knob — 1e-4 drives the adaptive loop to a
genuinely better POLE SET (fit rms 0.010 -> 0.0021) and the T level
FOLLOWS: T rms 0.0432 -> **0.02984** (new winner, tgt1e4/s2 + lead1
inherited, DOS 1.441, osc 0.0079, min eig -0.074; artifacts
cmp_fits_final_v2.npz / method_summary_final_v2.json /
tmp_fit_final_v2.gz). This REFINES finding (3): at a FIXED pole set
T is model-shape-limited (polish inert), but a tighter target
changes the ADAPTIVE trajectory and T follows. (b) targets <= 1e-4
are equivalent here (the two-stage floor's absolute 3e-3 threshold
dominates; tgt1e5 bit-identical to tgt1e4). (c) w_max optimum at
0.25 is SHARP (0.22 -> T 0.149, 0.28 -> 0.083). (d) n_init: 24 right
(12/48 much worse at T). (e) tail anchors matter (off -> 0.074).
(f) INERT on this system: init mode (NO-OP under psd=True — the
dispatch always uses the NNLS PSD dictionary, verified with
mode-specific starts), local_floor_frac 1.0, full-fit ridge 0.05,
prune_rel_tol 1e-2, n_res=0 (T 0.0499 vs 0.0499 baseline-without...
i.e. the 8-pole boost is not critical at this operating point).
Full tables: Data2/nogap_Ag5/README_gram_refined.md (experiment
rounds section), sweep_finer/results_exp{2,3}.json. T-rms
progression on this system (same 457-pt reference throughout): old
3.py 0.113 -> 0.068 -> 0.0492 -> 0.0432 -> 0.0298.

**spinpol_ag5 sweep round (2026-10-09 night)**: sweep_framework
applied to the MULTI-K system (2 leads x 2 spins): one GammaSweep
per (lead, ik) slice (framework extended with lead_label so the
caches are disjoint; 4 slices run as parallel background processes),
health-gated multi-start (3-6 starts; ~half the starts collapse to
rms ~0.21 basins here - the gate is mandatory), equalize_pole_counts
across all four slices before inject_fits (one common pole count
required by Zandpack's NumL helpers; verified asserted + full tail
ran cleanly). THREE findings:
(1) TETA MISMATCH dominated the stored spinpol numbers: they were
computed at teta=1e-2 against a 2e-2 sampling - the SAME fit scores
0.044 at 1e-2 and 0.0093 at MATCHED eta (DOS 7.97 vs 2.76). All
future spinpol comparisons must use teta=2e-2; the 6.3 table row is
marked superseded.
(2) The stored knob mult=2.0 was tuned under the mismatched metric:
at matched eta, mult=1.0 + w_max=0.3 (m1_w03, 6-start) WINS with
T rms **0.00824** (base/mult2: 0.0243; m1 alone: 0.0093; w03 alone:
0.0153; tight target did NOT help here, unlike nogap - 0.0207).
6.4 table updated.
(3) Off-window health: the production tail prints Lorentzian-Gamma
eigenvalues +-1.3e8 (huge cancelling residues, max|C| up to 5.6e7 -
common to ALL high-pole-count fits here incl. base); the decisive
metrics favor the winner: padded min eig -0.09..-0.14 (within the
gate) and complex-plane limsup ratios 2e-5..4e-4, vs BASE's padded
min eig **-342** on one slice. The winner is genuinely the safer
model.
Artifacts: Data2/spinpol_ag5/README_sweep.md (full record),
sweep_spin/ (fit caches + results_eval{,_matched}.json +
results_final.json + figures), cmp_fits_sweep.npz,
method_summary_sweep.json, tmp_fit_sweep.gz (full tail run).
Framework note: GammaSweep now takes lead_label for multi-k slices;
run one sweep per (lead, ik) in parallel processes.

Common to every system: `local_floor_frac=1.5`, residual boost
`n_res=8` (PSD-forced), `channel_tol=1e-4` when p > 32, ridge 0.05
(final solve), tail anchors on, and `find_correction` with the SAME
window as that system's own 3.py.

Knob rules of thumb (from the sweeps):

- **eta_floor_mult**: measure the sharpest resolvable feature of the
  sampled Gamma. If it is ~5 x eta wide, mult=5 removes
  resolution-limit ringing (zgnr, the metals). If the features are
  NARROWER than 5 x eta, mult=5 destroys the fit (agnr, fig2ag5,
  spinpol) — use 1-3.
- **w_max_raw**: 0.5 eV default. Tighten to 0.35 only if slow T
  undulation persists AND the system has no real feature wider than
  that (zgnr_spin's broad spin-polarized features need 0.5).
- **psd_force**: True by default (Gamma >= 0 everywhere). False when
  the sampled Gamma has a deep narrow notch (dips toward zero — the
  convex PSD solve cannot carve it) or when forcing visibly collapses
  the rms (broad-metal Gammas); in both measured cases the kept
  unconstrained model was PSD-compatible anyway (padded-grid
  minEig >= 0).
- **Pole budget**: adaptive + max 100 saturates on its own ("adding
  no longer pays"); target_rms is NOT the pole-count knob. The boost
  (n_res=8) buys more than any fixed pole count.
- **Sampling beats knobs**: where a fit misbehaves locally, densify
  the sampling there (zgnr window extension, RSSE E=0 and edge
  densification) rather than loosening width floors globally.

## 7. Known caveats

1. **Local minima and BLAS nondeterminism.** Identical inputs can land
   in different basins run-to-run (threaded reductions flip trf
   trajectories on knife-edge cases). If reproducibility matters, pin
   `w_min` and/or `n_poles` (fixed-count runs are far more stable) and
   keep `OMP_NUM_THREADS=1` — raising it does not even pay in wall
   time post-compression (4.11).
2. **Width-floor knife edge.** Sharp examples can be sensitive to the
   floor near their narrowest features (fig2ag5: rms 0.0026 at w_min=1e-3
   vs 0.037 at 1.13e-3). The two-stage dispatch handles the observed
   cases; exotic data may need an explicit w_min.
3. **PSD projection costs accuracy** (inter-pole cancellation is real
   structure for the fit). The total-fit λ_min is reported on every
   benchmark figure; decide from that, not from per-coefficient
   eigenvalues.
4. **AuBreakJunction is under-resolved** at its sampling (HANDOFF
   attempt-1 reached the same conclusion): its λ_min violation and
   residual floor are a data problem; more sampling points, not more
   poles, are the fix.
5. **rms metric mixing** in the comparison figure: attempt-1 numbers are
   their entry-filtered frob_relerr (|max|>1e-3), ours is full-matrix.
   Our metric is the harder one, so the comparison direction is safe,
   but like-for-like re-computation is possible if needed.
6. `fit(max_nfev=None)` uses 400 evaluations per intermediate
   optimization and 1500 for the final polish — tuned for the current
   examples; large p runs can be shortened at slight accuracy cost.
7. **The pole cap changes the adaptive trajectory, not monotonically in
   quality.** The loop's insertion granularity depends on the remaining
   budget, so cap-40 and cap-60 runs diverge around n≈37 and can end in
   different minima: fig2ag5 and spinpol L0 fit BETTER at cap 40
   (0.0026/0.0030, 0.0059) than at cap 60 (0.02+), while ag3au3,
   ag3au3_gap, aubj and smallrsse improve at cap 60 (e.g. ag3au3
   0.0068 → 0.0021). The merged `summary.json` keeps the better run per
   example; the `source` field says which. If one cap is adopted as the
   pipeline default, verify the affected examples. A related guard
   exists: the final ridge solve is redone without the ridge if the
   ridge costs more than 2x (observed on fig2ag5 L1 at high pole counts).

## 8. Open items

- Ag3Au3 / Ag3Au3_gap 'exact' pin (apply_method.py): RESOLVED BY
  MEASUREMENT (2026-10-10 remaining-systems round, 4.15): on the
  gradient-refined grids with the 3-start health gate, gram m1_w03
  BEATS the pinned exact on Ag3Au3 (T rms 0.00623 vs 0.00699) and is
  statistically tied on Ag3Au3_gap (0.00364 vs 0.00349) at ~5x less
  wall. Recommendation: DROP the pin (edit the CFG entries to
  tr_solver='gram', gram_cut=1e-9, eta_floor_mult=1.0,
  w_max_raw=0.3 and use multi-start); the winning fits are cached in
  Data2/Ag3Au3*/sweep_ag3au3*/ and equalized in cmp_fits_sweep.npz.
  Awaiting user sign-off to edit apply_method.py.
- Performance, remaining levers (post refit_scales + PSD-fusion
  rounds, 4.11/4.13): the profile leaders are now LAPACK-bound
  (trust-region eigh 23.5%, project_psd's batched eigh, the QR
  stack). Further gains are ALGORITHMIC, not kernel-level: (a) fewer
  ADMM iterations for candidate RANKING (currently 50/candidate,
  82 calls/fit), (b) test skipping the nnls scale refit during
  ranking (now cheap, low priority). Kernel shortlist items are
  exhausted (the gram_and_grad einsums measure 0.2 s - dead).

**gram performance re-profile (2026-10-09, validation/
12_gram_profile/)**: cProfile over the production gram fit_boosted
(Ag3Au3 lead1, control start; 110 s under profiler vs 100-116 s
clean; reproduces the spread c0 record bit-for-bit). With the
optimizer now cheap, the cost has MOVED OUT of fit_poles (26% of
the wall: varproj factors 17.8%, eigh 11.0%, select_step 0.5%):
**73% is the PSD machinery, 53% of the whole fit is refit_scales'
scipy nnls** (0.67 s/call x 84 adaptive-loop candidates on a
(729*451, M) full-channel design — the 4.11 compression never
covered the PSD path; the optimizer runs on ~19 of 729 channels),
ADMM 18.2%. Shortlist (documented, not implemented): compress the
RANKING path's scale-refit/ADMM data term to the optimizer's
channel basis (final refine stays exact; the PSD PROJECTION must
stay in the original channel basis — the constraint is
basis-dependent), possibly skip the scale refit during ranking,
a faster NNLS for the final refit, einsum reductions in
gram_and_grad.

**refit_scales optimized (2026-10-09, validation/13_refit_scales/)**:
the factored normal-equations route — the NNLS design's Gram
FACTORIZES exactly (G = Hadamard(Phi^T Phi, Cv^T Cv), h from one
matmul), so nnls runs on the Cholesky-factorized (M x M) system with
the SAME minimizer; no (729*451, M) design is ever built. KKT-guarded
(alpha checked against the true quadratic's gradient; any suspicion
falls back to the original full-design nnls), so it can only change
SPEED. Gates: unit alpha/objective agreement at machine precision
(700-1000x per call), both test suites ALL PASS, production gram
test 583 s -> 244 s (2.4x, final T rms 0.00772 — better than the
stored 0.0100), re-profile refit_scales 54% -> 5.3% (profiled wall
110 s -> 43 s). Per-START basins shift (fp-level ranking changes flip
adaptive decisions — the 4.13 basin chaos; the 3-start health gate
caught the new lead1 c0 catastrophic basin). Post-optimization
leaders: the ADMM (51.9%, project_psd 36.5% inside it), eigh 21.8%,
einsum 20.9% — next targets per the 12_gram_profile shortlist.

**PSD/ADMM numba fusion (2026-10-09, validation/14_psd_numba_fusion/)**:
shortlist items 1+2. Four `_njit(cache=True)` kernels (PSD Hermitize,
PSD eigen-reconstruction, the ADMM X-update RHS, over-relaxation)
plus once-per-solve reused iteration workspaces: project_psd drops
from ~5 array allocations per call to 2, the ADMM loop's
per-iteration temporaries are gone. EVERY kernel is BITWISE-identical
to its numpy reference (even the reconstruct k-reduction matches
einsum's loop order), and the full psd_solve_admm trajectory is
bit-identical to the pre-change module on real production inputs —
so the end-to-end production runs reproduce their records DIGIT FOR
DIGIT (no basin shifts; RSSE spot 0.00854132/45 identical, Ag3Au3
production T rms 0.007719 identical). Walls: production 6-fit test
244 s -> 204 s, profiled gram fit 42.8 -> 35.1 s; einsum bucket
9.0 -> 0.2 s (it was project_psd's reconstruction, NOT gram_and_grad
— shortlist item 3 dead). Cumulative performance rounds
(refit_scales + this): profiled gram fit 110 -> 35.1 s (3.1x),
production test 583 -> 204 s (2.9x), quality unchanged (T rms 0.00772
vs stored exact-era 0.0100). Remaining leaders are LAPACK-bound
(trust-region eigh 23.5%, project_psd's batched eigh, QR stack);
further gains must be algorithmic (fewer ranking ADMM iterations /
skip the scale refit during ranking), not kernel-level.
- Transport-level validation — DONE since the last revision for every
  Data2 system (dos_t_comparison.png / method_comparison.png, sections
  6.2/6.3). Remaining: the Data/ bench examples have Gamma-level
  figures only (their old pipelines are not wired for a T comparison
  in the bench).
- nk > 0 handling — DONE (spinpol_ag5 and zgnr_spin run per-(lead, ik)
  through apply_method.py / refine_loop).
- Optional joint PSD solve (cvxpy/factored L L*) if strict PSD at
  percent-level cost is ever required.
- Pole-cap sensitivity (caveat 7): consider a two-cap run
  (fit at cap 40 and 60, keep the better) as a built-in multi-restart,
  or make the insertion granularity cap-independent.
- Performance, not yet adopted (4.11): the numba basis kernel (numba
  0.67 works on this Python; only worth it if a re-profile shows
  per-call NumPy overhead dominating post-compression), the
  Gram-matrix LM replacing TRF (guide 2.4), and parallel two-stage
  floor dispatch (guide 2.5).
- The derivative-only refinement mode (4.8): DONE since the 2026-10-
  09/10 rounds — exercised end-to-end on nogap_Ag5, agnr, Fig2Ag5,
  zgnr, Ag3Au3 and Ag3Au3_gap (the recommended recipe,
  do_remove=False; the zgnr_spin attempt hit a pathological-basin
  final fit, see 4.15). Remaining: none for the mode itself.
- Per-system width policy: PARTLY AUTOMATED since the sweep rounds —
  sweep_framework measures eta_floor_mult/w_max per system and the
  cross-system default (m1_w03) is established (4.15/4.16). An
  automatic FIRST-GUESS selector (from the sharpest resolvable
  feature width vs eta) would remove the remaining sweep cost.
- Pathological-basin fits have no in-flight wall guard (zgnr_spin's
  refinement final fit >55 min; nogap's w30_exact s1) — a max-wall
  knob for refine_loop / the sweeps (stop a start, take the next)
  is the natural fix.
- AuBreakJunction: both pipelines remain limited by the under-resolved
  sampling (caveat 4); a denser contour would benefit it most.
