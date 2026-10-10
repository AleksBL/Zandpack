# Implementation Plan: Shared-Pole Lorentzian Fit of Hermitian Matrix-Valued Functions

## Goal

Given samples $F(E_n) \in \mathbb{C}^{p \times p}$, $F(E_n) = F(E_n)^*$, $n = 1,\dots,N$, fit

$$F(E) \;\approx\; \sum_{i=1}^{M} C_i\,\frac{w_i^2}{(E-e_i)^2 + w_i^2}, \qquad C_i = C_i^*,$$

with strictly proper decay $F(E) \to 0$ as $|E| \to \infty$ (satisfied identically by the basis), shared pole pairs $(e_i, w_i)$ across all matrix entries, and Hermitian (optionally PSD) residue matrices $C_i$.

## Architecture

```
input:  E[N], F[N,p,p]        (samples, Hermitian)
  │
  ├─ 1. Pole initialization (log-spaced or VF pass)
  ├─ 2. Variable projection loop
  │      ├─ inner: linear LS for {C_i} at fixed poles
  │      └─ outer: Levenberg–Marquardt over {e_i, w_i}
  ├─ 3. Prune (negligible ‖C_i‖, duplicates) + add poles at residual peaks
  └─ 4. Final constrained solve (Hermitian / PSD) + validation
output: e[M], w[M], C[M,p,p]
```

## Step-by-step plan

### 0. Data preparation
- Verify/assert `allclose(F[n], F[n].conj().T)`; if not, symmetrize: `F[n] = (F[n] + F[n].conj().T)/2`.
- Rescale energies to O(1): `E' = (E - E_mid)/E_scale` (poles and widths scale the same way). Fit in scaled units; unscale at the end. This is the single biggest conditioning win.
- No constant column anywhere: your basis has no offset by construction, matching the required decay.

### 1. Pole initialization
- Default: $M$ centers $e_i$ log-spaced over $[\min E, \max E]$, widths $w_i \approx 1$–$5\%$ of the band.
- Better: run a pole-relocation pass first.
  - Option A (library): `scikit-rf` `VectorFitting` on the stacked matrix entries; take the fitted conjugate pole pairs as starting $(e_i, w_i)$; discard its residues.
  - Option B (self-contained, ~60 lines): implement the SK iteration:
    1. basis $\Phi_{n,k} = 1/(E_n - \xi_k)$ with $\xi_k = e_k \pm \mathrm{i} w_k$;
    2. solve stacked LS for $r̃_k$, $\tilde C_k$ in $\sigma(E_n) F(E_n) \approx \sum_k \tilde C_k/(E_n-\xi_k)$, $\sigma = 1 + \sum_k r̃_k/(E_n-\xi_k)$ (enforce $\sigma \to 1$ asymptotically);
    3. new poles = zeros of $\sigma$ (companion-matrix eigenvalue), flip unstable ones to the correct half-plane, enforce conjugate pairing;
    4. iterate 3–10 times.
- Overinitialize: start with more poles than expected resonances; pruning (step 3) removes the excess.

### 2. Variable projection (core routine)
- `residuals(theta)`: given `theta = [e_1, w_1, ..., e_M, w_M]`:
  1. build `Phi[N, M]` with `Phi[n,i] = w_i**2 / ((E_n - e_i)**2 + w_i**2)` (real);
  2. solve the linear LS for the coefficients. Two layouts:
     - entrywise: for each upper-triangular (a,b), solve `min || Phi @ x - f_ab ||` with `f_ab[n] = F[n,a,b]` — real diagonal, complex off-diagonals handled via real/imag stacking;
     - vectorized: build one real design matrix (Khatri–Rao form) and solve all entries at once with `numpy.linalg.lstsq`.
  3. return the stacked residual vector; the optimal $C_i$ are re-solved fresh each call (never passed as outer variables).
- Outer optimization: `scipy.optimize.least_squares(residuals, theta0, method='lm' or 'trf')`. Use `trf` with bounds: $w_i > 0$, $e_i$ within or slightly beyond the energy range.
- Weights: to avoid the largest matrix entries dominating, weight per sample by $1/\max(\|F_n\|, \epsilon)$ for relative error. Decide up front absolute vs. relative error — it changes the chosen poles.

### 3. Prune and refine
- Prune: remove pair $i$ if $\|C_i\|_F$ is below a threshold relative to $\max_j \|C_j\|_F$, or if two pairs $(e_i, w_i) \approx (e_j, w_j)$ overlap (near-cancellation). Refit after pruning.
- Add: evaluate the residual $\|F_n - \hat F(E_n)\|_F$; if a sub-band exceeds tolerance, insert a new pair centered at the worst $E_n$ with width set to the local sample spacing; refit.
- Stop when target RMS/max error is met or an information criterion says adding poles no longer pays.

### 4. Final coefficient solve with structure constraints
- Hermitian: parametrize each $C_i$ by its upper triangle ($p(p+1)/2$ reals: real diagonal, complex off-diagonal pairs). Solve the real-stacked LS once at the final poles.
- PSD (if $F(E) \succeq 0$, e.g. spectral functions): convex for fixed poles. Solve with `cvxpy`:
  ```python
  # variables C[i] Hermitian PSD
  # per-sample: F[n] ≈ sum_i Phi[n,i] * C[i]
  # vectorized: vec(F[n]) = (Phi[n,:] ⊗ I) vec(C_stack)  → linear in the C[i]
  ```
  For $p \gtrsim 10$, use a factored form $C_i = L_i L_i^*$ with Gauss–Newton on $L_i$ instead of a generic SDP.
- If PSD is only approximately true in the data, fit unconstrained Hermitian first, then project: diagonalize each $C_i$, clip negative eigenvalues to zero, refit scales.

### 5. Validation
- Hermiticity: exact by construction; assert anyway.
- Decay: no constant term exists, but check $\limsup$ on a large semicircle in the complex plane numerically — the real check is that pruning removed cancelling pairs (a fit good on the real axis but large in the complex plane signals hidden cancellation).
- Analyticity: all poles off the real axis ($w_i > 0$); symmetric pair placement is the Herglotz configuration consistent with upper-half-plane analyticity.
- Hold out a few samples as a test set; report error on them, not just on fitted points.
- Kramers–Kronig sanity: if the underlying $F$ is analytic in the upper half-plane, the fit's dispersive part (same poles, principal-value basis) should match the data's dispersive part if available. A mismatch means the absorptive-only basis is absorbing structure it shouldn't.

## Caveats specific to this case

1. Absorptive-only basis. The Lorentzian basis carries no dispersive (principal-value) component. If the data contains background or asymmetric tails, the fit biases peak amplitudes. Test: refit with the full pole–residue basis (add $(E-e_i)/((E-e_i)^2+w_i^2)$ as a second column per pair) and compare; if dispersive coefficients come out large, the restricted basis is wrong for your data.
2. Decay is automatic per term but not the whole story. Uniform decay as $|E|\to\infty$ in the complex plane is violated by near-cancelling distant pairs even though each term decays. Pruning is the enforcement mechanism — do not skip it.
3. Ill-conditioning is intrinsic. Clustered poles or $w_i \ll$ sample spacing make `Phi` nearly rank-deficient. Mitigations: energy rescaling (step 0), column orthonormalization (QR) inside the LS solve, and refusing to rescale columns computed through massive cancellation. Do not interpret tiny singular values as "more physics".
4. PSD data vs. unconstrained fit. Unconstrained LS does not preserve positivity between samples; for spectral-function use the PSD constraint matters. Order of operations: get poles unconstrained, then do the PSD coefficient solve, then optionally one projected refinement pass. The joint problem is not convex; this decomposition is the practical workaround.
5. Variable projection is local. It converges to the nearest local minimum from the initialization. The VF/SK initialization (step 1) is what stands between you and bad local minima; for multimodal problems, try 2–3 restarts with different pole spreads and keep the best.
6. Complex arithmetic discipline. Work with real-stacked LS (real and imaginary parts as separate rows); Hermitian pairing halves the unknowns. Do not let a naive complex `lstsq` fit the lower triangle redundantly — it doubles cost and can mask symmetry-breaking noise.
7. Choice of error norm. Frobenius-over-all-entries is the default; per-entry weighting changes the selected poles. Decide based on downstream use.
8. Parameter count vs. samples. Unknowns: $2M$ pole reals + $M \cdot p(p+1)$ coefficient reals (Hermitian). You need $N \gg M$ per entry for a non-degenerate fit; the singular spectrum of `Phi` (or a Loewner-matrix SVD) gives a data-driven check on whether $M$ is supportable.

## Suggested module layout

```
lorentz_fit/
  data.py         # loading, Hermitization, scaling
  poles_init.py   # log-spaced init, SK/VF relocation pass
  varproj.py      # Phi construction, inner linear LS, residuals, outer least_squares
  constraints.py  # Hermitian parametrization, PSD solve (cvxpy), projection fallback
  prune.py        # residue-norm pruning, duplicate detection, residual-driven insertion
  validate.py     # held-out error, complex-plane limsup, KK cross-check
```

## External dependencies
- numpy, scipy (LS, `least_squares`)
- scikit-rf (optional, pole initialization only)
- cvxpy (only if PSD residues are required)