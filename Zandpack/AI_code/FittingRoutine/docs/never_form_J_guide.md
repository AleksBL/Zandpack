# "Never form J": a Gram-matrix trust-region step for the variable-projection fit

Companion to `varproj_factors_speedup_guide.md`, `varproj_next_steps.md` and
`lsmr_vs_exact_edge_band.md`. Based on the uploaded `AI_fit.py` (current state:
`tr_solver="lsmr"` default, `least_squares` always called with bounds).

Status: the identities, the gradient and the step agreement below were checked
numerically against your own `_assemble_dense_j`. A full vendored optimizer loop
was **not** written or run. Treat the speed numbers as indicative: they are
single-thread timings on synthetic Lorentzian problems.

## 1. Summary

scipy's `trf` needs the Jacobian only through two small objects:

- the Gram matrix `G = J^T J` (2M x 2M), and
- the gradient `g = J^T f` (length 2M).

Both can be built directly from the variable-projection factors you already
compute (`Z`, `X`, `S`, `R1`), at a cost independent of the number of channels.
The dense `J` (rows*n_ch x 2M) and the `VarProjJacobian` operator are then
never needed, and the per-iteration SVD (exact) or lsmr loop is replaced by one
symmetric eigendecomposition of a 2M x 2M matrix.

Measured (synthetic, single thread, R = rows incl. ridge, n = 2M):

| Problem | assemble J | `svd(J_aug)` | lsmr (dense J, default maxiter) | `G` build | `eigh(G)` |
|---|---|---|---|---|---|
| RSSE-like R=264, n=150 | 4.7 ms | 54.6 ms | 55.1 ms | 0.4 ms | 3.3 ms |
| ag3au3-like R=551, n=200 | 13.4 ms | 141.1 ms | 139.1 ms | 0.5 ms | 4.8 ms |
| nogap-like R=1071, n=140 | 20.8 ms | 186.5 ms | 168.8 ms | 0.5 ms | 2.8 ms |

lsmr ran its full n iterations in these tests. On real data it may stop earlier,
so re-measure on your fits.

## 2. The identities

Notation follows `_varproj_core` / `_varproj_jac`: `Res = A X - Y`,
`Z = (I - P_A) D`, `K = A (A^T A)^-1 = Q R1^-T`, `S = Z^T Res`, `k(p) = p // 2`
(the pole that parameter p belongs to). With `n_ch` the (compressed) channel count:

```
J[:, p]  =  vec( Z[:, p] x_k(p)^T  -  K[:, k(p)] S[p, :] )        # as in _jac_dense_kernel
```

### 2.1 Orthogonality removes the cross terms

`Z` lies in `range(A)^perp` and the columns of `K` lie in `range(A)`, so
`Z^T K = 0`. (Measured relative size: 1e-15.) Expanding `J^T J` entry by entry
the two cross terms therefore vanish, leaving a sum of two Hadamard products:

```
G[p, q] = (Z^T Z)[p, q] * (X X^T)[k(p), k(q)]  +  (K^T K)[k(p), k(q)] * (S S^T)[p, q]
```

### 2.2 `K^T K` needs no row sum

On the QR path `K = Q R1^-T`, so `K^T K = R1^-1 R1^-T` (an M x M matrix, O(M^3)).
Only `Z^T Z` costs O(R M^2). On the SVD-fallback path use `K.T @ K` instead
(`R1` does not exist there).

### 2.3 Gradient

```
g[p] = Z[:, p] . (Res X^T)[:, k(p)]  -  (K^T Res)[k(p), :] . S[p, :]
```

`K^T Res = R1^-1 (Q^T Res)` is about 1e-13 relative to `|Res|` because
`Res` is orthogonal to `range(A)` (Kaufman's simplification). Keep the term
anyway; it is cheap. Verified against `J.T @ Res.ravel()` to 1e-15, with
weights and ridge.

### 2.4 Column norms

`diag(G)` is exactly the `col2` you already compute for the `x_scale` of the
operator branch, so `compute_jac_scale` becomes `np.sqrt(np.diag(G))`.

## 3. Code

### 3.1 Gram and gradient from the factors

```python
import numpy as np
from scipy.linalg import solve_triangular

def gram_and_grad(Res, X, Z, S, R1=None, Q=None, K=None):
    """G = J^T J (2M x 2M) and g = J^T f without forming J.
    QR path: pass R1 and Q.  SVD-fallback path: pass K (R1 is None)."""
    M = X.shape[0]
    k = np.arange(2 * M) // 2

    ZZ = Z.T @ Z
    XX = X @ X.T
    SS = S @ S.T
    if R1 is not None:
        Ri = solve_triangular(R1, np.eye(M), check_finite=False)
        KK = Ri @ Ri.T                          # K^T K = R1^-1 R1^-T
        KtRes = solve_triangular(R1, Q.T @ Res, check_finite=False)
    else:
        KK = K.T @ K
        KtRes = K.T @ Res

    G = ZZ * XX[np.ix_(k, k)] + KK[np.ix_(k, k)] * SS
    G = 0.5 * (G + G.T)                          # enforce exact symmetry for eigh

    T = Res @ X.T                                # (R, M)
    g = np.einsum("np,np->p", Z, T[:, k])
    g -= np.einsum("pc,pc->p", S, KtRes[k])
    return G, g
```

`np.ix_` expansions can be replaced by a small numba loop (see section 7).
`K` itself is not needed on the QR path, so the `solve_triangular(R1, Q.T).T`
step in `_varproj_jac` can be skipped when running this solver.

### 3.2 What `trf_bounds` uses from J, and the replacements

In scipy's `trf_bounds` (scipy 1.17.1 here; check your version) J appears in:

| scipy usage | Replacement |
|---|---|
| `g = compute_grad(J, f)` | `g` from section 3.1 |
| `compute_jac_scale(J)` | `np.sqrt(np.diag(G))`, with scipy's own zero-guard |
| `J_augmented[:m] = J * d`, `svd(J_augmented)`, `uf = U.T @ f_augmented` | eigendecomposition block below |
| `select_step(x, J_h, ...)` and `build_quadratic_1d(J_h, ...)` (only form `J_h s`) | pass an n x n `Rg_h` with `Rg_h^T Rg_h = G_h`, e.g. `diag(sqrt(lam)) V^T * d` |
| cost `0.5 * f.f`, `f_new` | unchanged: always use the **true** full residual for accept/reject |

The eigendecomposition block, replacing the exact-path SVD:

```python
from scipy.linalg import eigh
from scipy.optimize._lsq.common import solve_lsq_trust_region   # private API

# per outer iteration, with d, diag_h, g_h as defined by trf_bounds
G_h   = G * np.outer(d, d)                 # J_h^T J_h
G_aug = G_h + np.diag(diag_h)              # + Coleman-Li term (augmented rows)
lam, V = eigh(G_aug, driver="evd")         # or np.linalg.eigh
lam, V = lam[::-1], V[:, ::-1]
s = np.sqrt(np.maximum(lam, 0.0))
keep = s > CUT * s[0]                      # CUT ~ 1e-7, see caveat 4.1
uf = (V[:, keep].T @ g_h) / s[keep]        # = U^T f_augmented
p_h, alpha, n_iter = solve_lsq_trust_region(
    int(keep.sum()), m_rows, uf, s[keep], V[:, keep], Delta,
    initial_alpha=alpha)
```

`m_rows` is the real J row count (only used by scipy for its rank threshold).
The Jacobian callback now returns `(G, g)` instead of a matrix. This cannot be
done through `least_squares`, so the solver loop has to be copied into your
package (about 150 lines for `trf_bounds`). Also copy `trf_no_bounds` if
`fit_poles` can be called without `bounds_fun`.

### 3.3 Integration in `fit_poles`

`fit_poles` already has the right structure: a cached core and lazily computed
Jacobian pieces. Replace `jac(theta)` with a function that returns
`gram_and_grad(...)` from the cached `(Res, X, Z, S, R1, Q)`. The
`operator_threshold` branch and `VarProjJacobian` disappear, along with
`_assemble_dense_j`, which removes the (rows*n_ch x 2M) memory peak.

## 4. Caveats

### 4.1 Conditioning: the main risk

Forming `J^T J` squares the condition number. `eigh(G)` resolves singular values
only down to about `sqrt(eps) ~ 1e-8` of the largest, whereas scipy's SVD path
goes to about `eps * max(m, n) ~ 1e-12`. Directions below the floor are
truncated by `CUT` instead of stepped along. The trust region normally clamps
those steps anyway, so the difference is small when the Jacobian is moderately
conditioned, but it is not zero.

Measured agreement of the trust-region step with scipy's exact SVD step
(`solve_lsq_trust_region`, `CUT = 1e-7`, Delta from 0.05x to 5x the Gauss-Newton
step):

| cond(J) | step relative difference |
|---|---|
| 1e3 | 6e-15 to 2e-12 |
| 1.4e4 | 3e-14 to 2e-10 |
| 4.3e4 | 4e-14 to 5e-9 |

These synthetic problems are mild. Real fits with near-coincident poles and a
small ridge can have much larger condition numbers. **Verify on real J before
trusting the agreement** (section 5).

Practical consequences:

- Choose `CUT` deliberately. About `sqrt(eps)` is where `eigh` stops being
  reliable, but it is a tunable. Report how many directions are truncated per
  iteration during validation.
- Cases where the exact solver takes large steps along weakly determined
  directions (up to the trust-region radius) can behave differently here.
- Do not use `G` to solve for the inner coefficients. The inner LS solve stays
  QR-based exactly as now.

### 4.2 Different rounding, so the bit-for-bit gate is gone

The trajectory will not be bit-identical to scipy's exact solver. Use the
relaxed gate: same iteration count and accept/reject pattern, final cost within
a pre-agreed tolerance, then the T-level metrics (bias and ringing split from
`lsmr_vs_exact_edge_band.md`).

### 4.3 The SVD-fallback path

When the QR rank check triggers, `R1` and `Q` are missing. Use `K.T @ K` and
`K.T @ Res` (section 3.1). `Z^T K = 0` still holds approximately because `K`
then comes from the truncated pseudo-inverse. The fallback fires in about 0.3%
of calls, so correctness matters more than speed here. Test it explicitly.

### 4.4 Private scipy API and maintenance

`solve_lsq_trust_region`, `select_step`, `build_quadratic_1d`, `update_tr_radius`,
`check_termination`, `CL_scaling_vector` and friends live in
`scipy.optimize._lsq` (private). A copied loop can drift from scipy's behavior
and break on a scipy upgrade. Pin the scipy version, keep the vendored file
next to a test that compares it with `least_squares(..., tr_solver="exact")` on
a small problem, and re-run that test after upgrades.

### 4.5 Details of the copied loop

- Keep scipy's termination logic (`ftol`, `xtol`, `gtol`) and `max_nfev`
  accounting unchanged so iteration counts stay comparable.
- The accept/reject test must use the true cost `0.5 * ||f||^2` from the full
  residual (ridge rows included). Only the step uses `G` and `g`.
- `loss_function` is unused in your code. Keep it that way, since a robust loss
  would reweight J rows and invalidate the Gram identity.
- `x_scale`: the dense branch uses `"jac"`, the operator branch a fixed vector
  from `col2`. With `G` both reduce to column norms of J.
- Symmetrize `G_aug`, and clip tiny negative eigenvalues to zero before the
  square root.

### 4.6 What this does not change

- The inner QR solve, the SVD fallback and the ridge construction are untouched.
- The cost-only (trial) evaluations are unchanged.
- Channel compression becomes less important for the solver, because `n_ch` enters
  `G` only through `X X^T`, `S S^T` and `S`. It still reduces the work in `Q^T Y`
  and `Res`. Once the Gram solver is validated you could test whether turning
  compression off removes its truncation error at little cost.

## 5. Validation plan

1. **Dump real data.** In a real fit (RSSE and ag3au3, early and late
   iterations), save `Res, X, Z, K, S, R1, Q`, the dense `J` and `f`.
2. **Identities.** `|G - J^T J| / |J^T J|` and `|g - J^T f| / |g|` should be at
   the 1e-15 level on the QR path.
3. **Conditioning.** Record `cond(J)`, the singular-value spectrum, and how many
   directions fall below `CUT`.
4. **Step agreement.** For several `Delta`, compare the Gram step against
   `solve_lsq_trust_region` on the SVD of `J_augmented`, including with active
   bounds (nonzero `diag_h`).
5. **Fallback path.** Force an SVD-fallback case and repeat 2 and 4.
6. **Trajectory.** Full fit with the vendored solver versus
   `least_squares(tr_solver="exact")`: compare `nfev`, `njev`, accepted/rejected
   pattern, final cost, fitted poles.
7. **Physics metrics.** T-level bias, ringing, E=0 notch and the integrated
   check from `lsmr_vs_exact_edge_band.md`.
8. **Timing.** 3 to 5 alternating runs, `OMP_NUM_THREADS=1`, medians, on real
   systems. Re-profile afterwards: the remaining time will be in the factors
   and in `eigh`.

## 6. Expected gain (an estimate)

In a synthetic end-to-end fit (p=4, N=300, exact solver) 60% of the time was in
`svd`, 14% in `_assemble_dense_j` and 4% in `compute_jac_scale`. All of that is
replaced by the Gram path. On your RSSE profile (SVD 66.5%, `_varproj_*` 3.8%) a
reduction of the solver part by roughly 10x would leave the fit at about half of
the exact-solver time. The real figure depends on what the remaining ~30% is,
and on how early lsmr stops on your data, so measure before committing.

If the Gram solver follows the exact trajectory closely, it makes the lsmr
default unnecessary and removes the edge-band bias versus ringing trade-off. Check
that against the T-level metrics.

## 7. What remains per iteration, and numba

Component timings after the Gram path (single thread, ms per call):

| Component | RSSE-like | ag3au3-like | nogap-like |
|---|---|---|---|
| basis + weights + ridge pad (trial) | 0.10 | 0.29 | 0.45 |
| QR (trial) | 0.65 | 2.79 | 3.29 |
| solve X, Res (trial) | 0.12 | 0.25 | 0.20 |
| D build (accepted) | 0.14 | 0.88 | 1.35 |
| Z = D - Q Q^T D | 0.33 | 1.06 | 1.25 |
| K triangular solve | 0.23 | 0.75 | 0.75 |
| `G` build | 0.62 | 1.42 | 0.99 |
| `eigh(G)`, scipy default | 3.0 | 5.8 | 3.2 |

Most of it is LAPACK/BLAS (QR, `eigh`, GEMM), which numba cannot beat. What is
worth doing, in order of payoff:

1. **`eigh` driver.** `np.linalg.eigh` or `scipy.linalg.eigh(G, driver="evd")` was
   25 to 40% faster than scipy's default (2.3 vs 3.0, 4.0 vs 5.8, 2.0 vs 3.2 ms).
   No numba needed, and it is the largest remaining item.
2. **Skip `K`.** The Gram path needs only `R1` (see 3.1), so the K solve
   (0.2 to 0.8 ms) can be dropped.
3. **Fused numba kernels** for the basis and derivative designs. Checked against
   the module output (max difference 1e-15):

   | Kernel | numpy | numba |
   |---|---|---|
   | A fill with weights and ridge rows | 0.10 / 0.29 / 0.45 ms | 0.02 / 0.07 / 0.10 ms |
   | D fill (interleaved, weighted) | 0.14 / 0.88 / 1.35 ms | 0.07 / 0.11 / 0.17 ms |
   | `G` expansion (`np.ix_` loop) | 0.16 to 0.30 ms | 0.02 to 0.04 ms |

   ```python
   from numba import njit

   @njit(cache=True)
   def fill_A(E, e, w, sqrt_sw, ridge, A):          # A: (N+M, M), zeros on entry
       N, M = E.shape[0], e.shape[0]
       for i in range(M):
           w2 = w[i] * w[i]
           for n in range(N):
               d = E[n] - e[i]
               A[n, i] = sqrt_sw[n] * w2 / (d * d + w2)
           if ridge > 0.0:
               A[N + i, i] = ridge

   @njit(cache=True)
   def fill_D(E, e, w, sqrt_sw, D):                 # D: (N+M, 2M), zeros on entry
       N, M = E.shape[0], e.shape[0]
       for i in range(M):
           w2 = w[i] * w[i]
           for n in range(N):
               d = E[n] - e[i]
               Dn = d * d + w2
               D2 = Dn * Dn
               D[n, 2 * i]     = sqrt_sw[n] * 2.0 * w2 * d / D2
               D[n, 2 * i + 1] = sqrt_sw[n] * 2.0 * w[i] * d * d / D2
   ```

   Pass `sqrt(sample_weights)`, as `_weighted_design` does. Keep these serial
   (`prange` overhead exceeds the work at 45k elements) and keep the existing
   `_njit is None` fallback. Total saving is about 0.3 to 1.5 ms per accepted
   iteration out of roughly 8 to 12 ms, i.e. 5 to 10% of what remains.
4. **Hoist constants.** `Y = B.T * sqrt(sw)` and its zero-padded ridge rows are
   rebuilt on every `_varproj_core` call but are constant per `fit_poles` call.
   Build once, outside `fun`. Small, free and bit-identical.
5. **Optional, with care:** `Z^T Z = D^T D - (Q^T D)^T (Q^T D)` avoids forming `Z`
   but cancels badly when the derivative columns lie mostly in `range(A)` (broad
   poles). Only try it if `Z` shows up in the profile, and compare the small
   eigenvalues of `G` against the explicit-`Z` version.

`dsyrk` for `Z^T Z` was no faster than a plain GEMM at these sizes (0.17 vs 0.16,
0.61 vs 0.52, 0.70 vs 0.60 ms), so skip it.

## 8. Order of work

1. Dump real `J`, `f` and factors (section 5, steps 1 to 5). If the identities
   or step agreement fail on real data, stop here.
2. Vendor `trf_bounds` as `trf_gram` and test it against
   `least_squares(tr_solver="exact")` on a small synthetic problem.
3. Run the full-fit comparison and the T-level metrics (steps 6 and 7).
4. Switch the `fit_poles` default only if the metrics match the exact solver; keep
   `tr_solver` as an escape hatch (`"exact"`, `"lsmr"`, `"gram"`).
5. Then the cheap items of section 7 (`eigh` driver, skip K, hoist Y), and numba
   kernels last.
