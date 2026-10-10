# 10_gram_solver — the "never form J" Gram trust-region solver

Implementation of `never_form_J_guide.md` (2026-10-09): a vendored
scipy `trf_bounds` loop (`AI_fit._trf_gram`) where the Jacobian
callback returns `(G, g) = (J^T J, J^T f)` built directly from the
variable-projection factors (`AI_fit.gram_and_grad`), plus the
`tr_solver="gram"` branch in `AI_fit.fit_poles` (with the K-solve
skipped on the QR path, `_varproj_jac(..., with_k=False)`). The
per-iteration SVD of the augmented Jacobian (scipy 'exact') or the
lsmr inner loop is replaced by one `np.linalg.eigh` of the augmented
Gram matrix; weak directions below `gram_cut` (default 1e-7) *
s[0] are truncated.

Standing rules apply (OMP_NUM_THREADS=1; pickles before sys.path).
The default solver is still 'lsmr'; 'gram' is opt-in via
`fit(..., tr_solver="gram", gram_cut=1e-7)`.

## Scripts (guide section 5 validation plan, in order)

| script | guide step | what it verifies |
|---|---|---|
| identities.py | 1-2, 4.3 | G = J^T J and g = J^T f from the factors vs the dense J (AI_fit._assemble_dense_j), on synthetic RSSE/ag3au3/nogap-shaped problems AND real VerySmallRSSE data, early + late iterates, QR path AND forced SVD fallback; also eig(G) vs sv(J) spectra and cond(J)/truncation counts |
| smoke.py | 8.2 | fit_poles exact vs lsmr vs gram on a small synthetic: cost/nfev/status |
| step_agreement.py | 5 (steps 4-5) | the trust-region step (several Delta, 0.05x-5x GN) and the FULL select_step (reflection/anti-gradient) vs scipy's own exact path on real RSSE factors with active bounds; both factor paths |
| trajectory.py | 5 (steps 6-8) | full fit_poles trajectories on real RSSE (both leads) + production fit_boosted walls, 3 alternating reps each, exact/lsmr/gram |
| ag3au3_gate.py | 5 (steps 6-7) | the heavy pinned-to-exact system: full production fit_boosted exact vs gram (wall, rms, poles, dense-grid model dev, PSD min eig) |

Plus the permanent regression guards in `tests/AI_fit_test_synthetic.py`
(`check_gram_solver`): identities on both paths + a fit_poles
exact-vs-gram trajectory gate on a small synthetic problem. RERUN
THOSE after any scipy upgrade (the vendored loop follows scipy
1.18.1's `trf_bounds`; the private API is
`scipy.optimize._lsq.common`).

## Results (2026-10-09, OMP_NUM_THREADS=1)

- identities: G/g errors 4e-16 .. 5e-15 everywhere (QR and SVD
  fallback, synthetic + real); eig(G) == sv(J)^2 to 1e-14 (one
  1.8e-8 at a cond(J)=5e15 fallback case - at the sqrt(eps) floor
  the guide predicts).
- step agreement: worst 2.5e-14 over all Delta/leads/paths (the
  guide's own synthetic worst was 5e-9); select_step agreement
  4e-19; zero truncated directions on real RSSE factors.
- trajectories: RSSE lead1 gram follows exact EXACTLY (nfev 194 =
  194, cost to 8 digits, poles to 1e-10); lead0 (max_nfev-capped,
  still descending): gram slightly DEEPER cost (3.7394e-4 vs
  3.7411e-4), poles within 2e-2, wall 0.82s vs exact 6.67s (8x).
- production fit_boosted (RSSE, medians of 3): exact 27.3s /
  rms 0.0114 / n=50; lsmr 18.6s / 0.00103 / 67; gram 9.2s / 0.0085 /
  45 - 3x faster than exact, 2x faster than lsmr; the adaptive loop
  is basin-sensitive for every solver (same phenomenon as the
  lsmr-vs-exact round; per-system quality must be checked, not
  assumed).
- synthetic gate (tests/): gram reaches the exact-solver minimum to
  4.8e-14 relative cost.

## ag3au3 gate + cut sweep (the system pinned to 'exact')

Full production fit_boosted, Ag3Au3 lead1 (451 pts, p=27, n_ch 729 ->
compressed 19):

| solver | wall | sample rms | n poles | dense-grid min eig |
|---|---|---|---|---|
| exact | 342.9 s | 0.0149 | 77 | -1.7e-2 |
| gram, cut=1e-7 (old default) | 40.2 s | 0.0276 | 93 | **-1.6e5** |
| gram, cut=1e-9 (NEW default) | 45.6 s | 0.0228 | 99 | -3.2e-2 |
| gram, cut=1e-11 | 49.6 s | 0.0228 | 99 | -3.2e-2 |

Identity check on REAL ag3au3 factors (late-ish iterate): G err
6.5e-16, g err 3.8e-14, cond(J) 1.4e4, ZERO truncated directions at
either cut - the fit_poles-level math is faithful. The production
difference is ADAPTIVE-LOOP BASIN DIVERGENCE (the same phenomenon
that made lsmr land non-PSD basins on this system): small step
differences early change prune/insert decisions and the loop
amplifies them. The cut matters because ag3au3 has genuinely weak
directions (s ~ 1e-8 s[0]) that 1e-7 wrongly truncates; 1e-9 and
1e-11 agree bit-for-bit there.

Conclusion adopted: gram_cut DEFAULT is 1e-9 (see HANDOFF 4.11);
Ag3Au3 stays PINNED to tr_solver='exact' (gram is 7.6x faster but
lands a different, slightly worse basin; sample rms 0.0228 vs
0.0149); 'gram' remains opt-in everywhere.

## Caveats (guide 4.1/4.2, confirmed)

- NOT bit-identical to 'exact': weak directions below gram_cut are
  truncated (eigh resolves only to ~sqrt(eps) of s[0]); use the
  relaxed gate (same nfev, same-quality cost, close poles).
- cond(J) > ~1e14 fallback cases sit at the eigh resolution floor
  (measured: RSSE lead1 SVD-fallback, 1/48 directions truncated) -
  the trust region clamps those directions anyway, but check
  `n_trunc` on the result for diagnostics.
- scipy private API (scipy.optimize._lsq.common), pinned 1.18.1.
