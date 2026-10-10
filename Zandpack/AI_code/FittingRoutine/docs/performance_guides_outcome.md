# Outcome of the lorentz_fit guides: what worked, what did not

Summary of the three guide documents (now in `.junk/`:
`lorentz_fit_review.md`, `lorentz_fit_performance_guide.md`,
`lorentz_fit_performance_guide_2.md`) — what was implemented, with what
measured effect, and what was rejected. Everything here is also in
HANDOFF.md sections 4.9 and 4.11.

## 1. lorentz_fit_review.md (drove the oscillation fixes, HANDOFF 4.9)

**Worked (implemented, measured):**
- Quadrature objective (4.1): sample weights carry trapezoid widths;
  all model-selection metrics use the quadrature rms. Required for
  refined (non-uniform) grids.
- eta_raw width floor (4.3): poles narrower than the data's imaginary
  broadening are forbidden. The multiplier became the most important
  per-system knob: eta_floor_mult=5 cured zgnr's resolution-limit
  ringing (sharp-feature T 0.64 -> 0.815) — but is catastrophic on
  systems with features narrower than 5*eta (agnr: relrms 0.52 at mult
  5 vs 0.0145 at mult 1; fig2ag5/spinpol destroyed in the suite
  check). Tune per system.
- psd_dict_init (4.7): NNLS non-negative Lorentzian dictionary fit of
  Tr Gamma as the PSD pole init — sparse, positivity-native.
- psd_solve_admm (4.6a): exact convex PSD coefficient solve at fixed
  poles, warm-started. PSD-projected candidate tracking + PSD
  insertion pass (B2/B3) — the fix for the PSD-forced slow undulation.
- Pole-pair symmetry question: confirmed numerically to 1e-16.

**Did not work (tried, measured, rejected):**
- True-relative weights 1/||F||^2 (4.4): degraded peak fidelity on
  data spanning ~4 orders of magnitude and broke the exact-recovery
  tests. Kept the half-relative form.
- Unconstrained residual fits in the boost path: broke positivity
  (min eigenvalue -0.84) and the oscillations returned. The correction
  fit must be PSD-forced.

**Not adopted (deferred):** dense-grid validation for model selection
(4.2), edge-continuation anchors (4.5), factored L L* joint
optimization (4.6c), committee-driven refinement (4.11).

## 2. lorentz_fit_performance_guide.md (drove HANDOFF 4.11)

The guide's own evidence base was small (p=6, dense-Jacobian path,
single core). Profiling the REAL regime (p=64, 451 samples, PSD) showed
a different dominant path: TRF/LSMR over a rank-structured
LinearOperator (74% of time), with the PSD path second (25%). The
guide's items still transferred well.

**Worked (implemented, measured):**
- 4.1 Channel compression — the decisive item. Real Ag3Au3 Gammas have
  numerical channel rank 9-19 of 729 at 1e-9-1e-4 relative; the
  optimizer runs on the compressed rows (trajectory unchanged — the
  discarded directions carry a theta-independent residual) and the
  final solve is always against the full responses. Synthetic profile
  case: 459 s -> 81 s from this alone.
- 4.3 Factor cache (fun/jac called at the same point): exact, small
  but free.
- 4.7 PSD path rewrites: batched (stacked-LAPACK) project_psd
  (75,672 Python-level eigh calls eliminated), vectorized channel
  pack/unpack (the ADMM looped over all channels twice per
  iteration), one Cholesky factor for all ADMM iterations. Together
  with 4.1's preconditioning of the problem sizes: profile 381 s
  after the first round.
- 4.8 Small items: basis_apply matmul replacing every
  einsum('ni,ijk->njk') (per-candidate evaluations), triangular
  solve in the QR path.
- 4.2 Vectorized dense Jacobian: implemented; matters on the dense
  path (small systems), no effect on the large-p hot path.
- 4.4 Tolerances (adopted via guide 2 with its refinement, below).

**Did not work / not adopted:**
- 4.5 Leaner candidate selection (slack preference for smaller pole
  sets): no measured gain in the guide's own benchmark, and it changes
  the selection semantics the tuned basins depend on. Rejected.
- 4.6 Warm-started second floor stage: deferred — basin risk (stage 2
  may reproduce stage 1's minimum); only pays when stage 2 triggers.
- 4.9 skrf init on compressed channels: unclear benefit over the
  existing port cap; not profiled as a hotspot. Rejected.

## 3. lorentz_fit_performance_guide_2.md

**Worked (implemented, measured, verified):**
- 2.1 Compress once per _fit_impl (B_opt passthrough): every internal
  fit_poles call reuses the compressed responses.
- 2.3 ADMM over-relaxation alpha=1.6 with tolerance 1e-6: verified
  EXACT — 200 iterations reach precisely the 5000-iteration reference
  objective (relative gap 0.0), solution PSD. Also correct not to
  compress the ADMM channel space (the guide's own warning: PSD
  feasibility may need out-of-span components).
- 2.2 Loose intermediate tolerances (1e-6) with a tight polish: the
  intermediate refits only seed candidates.

**Did not work / not adopted:**
- 2.4 Gram-matrix LM replacing TRF: invasive, changes convergence
  semantics; post-compression the dense SVD it targets is no longer
  leading. Rejected.
- 2.5 Parallel two-stage dispatch: wastes CPU whenever stage 1
  suffices (the common case). Rejected.
- 2.6 Partial refits of inserted pairs / coarse-to-fine grids: risky
  for narrow features, marginal gain. Rejected.
- Numba basis kernel (section 3): numba 0.67 does work on this
  Python 3.14, but after compression the basis arrays are small and
  LAPACK dominates elsewhere; only worth revisiting if a re-profile
  shows per-call NumPy overhead leading. Rejected for now.

## Net measured effect

| case | before | after | factor |
|---|---|---|---|
| synthetic p=64 PSD profile case | 459 s | 53 s | 8.7x |
| real Ag3Au3 lead1 fit (worst system) | ~1900 s | 537 s | 3.5x |
| Ag3Au3 full old-vs-new sweep | 3848 s | 963 s (Ag3Au3_gap) | ~4x |
| test suite | 11.1 s | 9.3 s | 1.2x |

Threading: OMP_NUM_THREADS=8 measured SLOWER than 1 (1068 s vs 782 s
on the real fit) — post-compression matrices are too small to amortize
thread overhead, and threaded BLAS breaks run-to-run reproducibility.
Keep 1.
