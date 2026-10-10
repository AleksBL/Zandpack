# 13_refit_scales — the factored normal-equations NNLS

`refit_scales` was 53% of the production gram fit (validation/
12_gram_profile): scipy's Lawson-Hanson `nnls` on the FULL design
(n_ch*N = 729*451 rows x M, up to ~260 MB materialized per call,
0.67 s per solve, ~84 calls per adaptive-loop fit).

## The optimization (AI_fit.py; backup backups/AI_fit_2026-10-09_pre_refitscales.py)

The design `A[(c,n), i] = Cv[c,i] * Phi[n,i]` is never formed. Its
normal equations FACTORIZE exactly:

    G[i,j] = sum_{c,n} Cv[c,i]Cv[c,j] Phi[n,i]Phi[n,j]
           = (Cv^T Cv)[i,j] * (Phi^T Phi)[i,j]      (Hadamard product)
    h[i]   = sum_c Cv[c,i] * (B @ Phi)[c,i]

and with G = L L^T (Cholesky), ||A a - b||^2 = ||L^T a - L^{-1} h||^2
+ const, so `nnls(L^T, L^{-1} h)` on the (M x M) triangular system
has EXACTLY the same minimizer as `nnls(A, b)` — the same convex
problem, computed at O((N + n_ch) M^2) with no large temporary.
The fast path (`_refit_scales_fast`) is KKT-checked against the TRUE
quadratic (w = h - G a: ~0 on the support, <= 0 off it); Cholesky
failure, non-finite output, or a KKT violation return None and the
caller falls back to the original full-design nnls — the fast path
can only change the SPEED, never the accepted solution quality.

## Gates (unit_gate.py, ALL PASS)

| case | alpha rel diff | objective rel diff | wall old -> new |
|---|---|---|---|
| synthetic M=24..100, n_ch=729 (rand) | 7e-15 .. 7e-14 | <= 2.4e-15 | 0.09-3.2 s -> 1-4 ms (100-1000x) |
| synthetic coincident-pole (degenerate) | 1e-14 .. 4e-14 | <= 6e-15 | 100-320x |
| real exactera l0 (M=108) | 3.9e-14 | 1.4e-16 | 3.5 s -> 4 ms |
| real exactera l1 (M=94) | 2.9e-07* | 1.9e-16 | 3.1 s -> 3 ms |
| real gram_production picks (M=107/98) | 1.5e-14, 2.8e-14 | 0 | 3.1-3.5 s -> 4 ms |

*the l1 2.9e-7 sits on a flat direction: both alphas achieve the
same objective to machine precision (1.9e-16) — a vertex-choice
difference in a degenerate direction, not an error.

Real cases use the production packing (scaled units, tail anchors,
the fits' true pole counts). PASS bar: alpha <= 1e-6 AND objective
<= 1e-9, or fallback-to-exact (the fallback never fired on these).

## End-to-end gates

- tests/AI_fit_test_synthetic.py + AI_fit_test_model_form.py:
  ALL PASS on the new code.
- Ag3Au3 gram production test (6 fits, 3 starts/lead, health gate):
  wall 583 s -> 244 s (2.4x); final T rms 0.00772 (pre-optimization
  run: 0.00937; stored exact-era: 0.0100; old pipeline: 0.0383);
  pair min eig -0.15. The per-START basins shifted (e.g. lead1 c0:
  rms 0.0198/-0.54 before -> 0.0185/-609 after) — EXPECTED and
  consistent with the basin-chaos finding (4.13): fp-level changes in
  the PSD ranking values flip adaptive-loop decisions; the 3-start
  health gate caught the new catastrophic basins (lead1 c0 and s2
  rejected) and the final pair is healthy. Pre-optimization
  artifacts preserved as *_pre_refitscales.* in
  validation/11_gram_basin_spread/.
- Re-profile (validation/12_gram_profile, same c0 start):
  refit_scales 59.2 s (54%) -> 2.3 s (5.3%); nnls 57.5 s -> 2.1 s;
  profiled fit wall 110 s -> 43 s (2.6x). New leaders: the ADMM
  (psd_solve_admm 51.9% + project_psd 36.5% inside it), eigh 21.8%,
  einsum 20.9% (gram_and_grad g-assembly + project_psd).

- Cross-system spot check (rsse_spot_check.py): RSSE lead0 gram c0
  reproduces the pre-optimization spread record to 1e-7
  (rms 0.00854132/45 poles vs 0.008541/45) — no knife-edge decisions
  crossed on that system, wall 8 s -> 6.8 s. The basin shifts are
  Ag3Au3-specific (its accept/reject decisions sit on finer edges).

## How to rerun

```
OMP_NUM_THREADS=1 python3 unit_gate.py              # ~1 min
OMP_NUM_THREADS=1 python3 ../../tests/AI_fit_test_synthetic.py
OMP_NUM_THREADS=1 python3 ../11_gram_basin_spread/ag3au3_gram_production.py
OMP_NUM_THREADS=1 python3 ../12_gram_profile/gram_profile_ag3au3.py
OMP_NUM_THREADS=1 python3 ../12_gram_profile/analyze_prof.py
```
