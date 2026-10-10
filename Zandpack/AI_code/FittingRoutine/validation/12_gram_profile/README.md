# 12_gram_profile — where the time goes in a production gram fit

cProfile over a full production `fit_boosted` (Ag3Au3 lead1, p=27,
n_ch=729 unfiltered -> optimizer-compressed to ~19, N=451,
`tr_solver='gram', gram_cut=1e-9`, psd=True/psd_force=False, control
start, adaptive base fit + 8-pole boost; the representative spread/
production calculation). Result reproduces the spread c0 record
bit-for-bit (rms 0.01977/92). Wall under profiler 110 s vs ~100-116 s
unprofiled — cProfile overhead is small; the C-call timings (nnls,
eigh, qr) are clean, Python-heavy paths are mildly inflated.

Files: `gram_profile_ag3au3.py` (runs + dumps `ag3au3_gram_c0.prof`),
`analyze_prof.py` (breakdown from the .prof), `inspect_prof.py`.

## The percentages (of the whole fit_boosted, cumulative)

| component | time | % | calls | note |
|---|---|---|---|---|
| **PSD machinery (psd_refine path)** | **79.7 s** | **72.7%** | 84 | per adaptive-loop candidate |
| — refit_scales nnls | 57.5 s | 52.8% | 86 | scipy Lawson-Hanson nnls, 0.67 s per call |
| — psd_solve_admm | 19.9 s | 18.2% | 84 | warm-started, 50-iter ranking budget |
| — project_psd (batched clip) | 13.5 s | 12.4% | 3814 | ~1 call per ADMM iter per batch |
| **fit_poles (the gram optimizer)** | **28.6 s** | **26.1%** | 78 | the round-10 target is now the MINORITY |
| — varproj factors (core QR+X/Res 7.4%, jac D/Z/S 4.6%, gram_and_grad 4.6%, dPhi 1.1%) | 19.5 s | 17.8% | 7-8k | LAPACK-bound, already optimized |
| — np.linalg.eigh (trust-region subproblem) | 12.0 s | 11.0% | 11138 | ~143 per fit_poles call |
| — _gram_select_step + quadratics | 0.5 s | 0.5% | 18.7k | negligible |
| everything else (basis, rms evals, solve_coefficients, prune/insert/merge, prepare, compress) | ~3 s | ~3% | | |

(Percentages sum past 100 for the PSD rows because project_psd runs
inside both psd_refine and the ADMM.)

Notable per-call facts: `_varproj_core` 8282 calls (one per theta
evaluation, ~1 ms each: qr 5.2 s + triangular solves), einsum 33.4k
calls / 7.9 s (the g-assembly reductions in `gram_and_grad`),
`unpack_coefficients` 3.1 s, channel-index generator expressions
~9M calls / 0.7 s.

## Interpretation

The gram round made the optimizer so cheap that the fit's cost
moved OUT of `fit_poles` entirely: **73% of the wall is the PSD
ranking/refine machinery**, and 53% of everything is a single
routine — `refit_scales`' `scipy.optimize.nnls`. Why it is expensive
at this problem size:

- refit_scales builds the FULL design `A` of shape (n_ch * N, M) =
  (729 * 451, up to ~100) = up to 33M doubles (~260 MB) per call and
  hands it to Lawson-Hanson active-set nnls — 0.67 s per solve,
  and the adaptive loop calls it ~84 times per fit (once per
  candidate ranking + final refines).
- The optimizer, by contrast, runs on the channel-COMPRESSED rows
  (~19 of 729) — the compression round (4.11) never touched the
  PSD scale refit or the ADMM, so they still pay full channel count.

## Optimization shortlist (documented, NOT yet implemented)

1. **Compress the ranking path's refit_scales/ADMM inputs** to the
   same compressed channel basis the optimizer uses (keep the FINAL
   refine exact). Same justification as the 4.11 compression: an
   orthogonal channel rotation leaves the LS invariant up to the
   discarded directions (rank ~19 of 729 at 1e-9). Expected: nnls
   rows 729->~19 (~38x) and the 260 MB design -> ~7 MB; the 57.5 s
   block should collapse to low single digits. CAVEAT: the PSD
   CONSTRAINT itself is channel-basis-dependent (project_psd works
   on the p x p matrices), so only the scale-refit LS and the ADMM's
   data term can be rotated — the projections must stay in the
   original channel basis, or the ranking must accept the rotated
   approximation while the final refine stays exact.
2. **Skip the nnls scale refit during candidate RANKING** (clip +
   ADMM-50 may order candidates equally well) — measure before
   adopting.
3. **Faster NNLS for the exact final refit** (Lawson-Hanson is slow
   at this size): scipy.optimize.lsq_linear(bounds) or a
   projected-gradient NNLS — only worth it after (1).
4. Minor: einsum -> (Z*T).sum(axis=0)-style reductions in
   gram_and_grad (7.9 s -> est. 2-3 s); channel-index genexprs in
   pack_responses (0.7 s).

Expected total effect of (1)+(2): ~60-70 s of the ~100-116 s
production wall; the optimizer (26%) and eigh (11%) become the
floor, together with ~20 s of irreducible ADMM.

## Optimization shortlist status (2026-10-09, after refit_scales round)

Item 1+3 are DONE via a different (simpler, exact) route — see
validation/13_refit_scales/: the NNLS design's normal equations
FACTORIZE (G = Hadamard(Phi^T Phi, Cv^T Cv), h from one matmul), so
the (n_ch*N, M) design is never formed; nnls runs on the Cholesky-
factorized (M x M) system with the SAME minimizer, KKT-guarded with
a full-design fallback. Measured: refit_scales 59.2 s (54% of the
fit) -> 2.3 s (5.3%); production gram fit 110 s -> 43 s profiled
(583 s -> 244 s for the 6-fit production test, 2.4x). The re-profile
(ag3au3_gram_c0.prof, post-optimization) shows the new leaders:

| component | % of fit (post) |
|---|---|
| psd_solve_admm | 51.9% (project_psd 36.5% inside it) |
| fit_poles optimizer | 38.3% (eigh 21.8%, varproj factors 26%*) |
| refit_scales | 5.3% (was 54%) |
| einsum (gram_and_grad g-assembly + project_psd) | 20.9% |

Remaining shortlist: (2) test skipping the scale refit during
candidate RANKING (now cheap — low priority); (4) einsum reductions
in gram_and_grad; NEW top target: the ADMM/project_psd pair (88% of
what remains of the PSD machinery) — channel-compression of the ADMM
data term is constrained by the PSD projection being channel-basis-
dependent; a cheaper project_psd or fewer ADMM iterations for
ranking are the angles.

*(overlapping cumulative shares; see analyze_prof.py output)*

## How to rerun

```
OMP_NUM_THREADS=1 python3 gram_profile_ag3au3.py   # ~4 min, dumps .prof
OMP_NUM_THREADS=1 python3 analyze_prof.py           # instant, from .prof
```
