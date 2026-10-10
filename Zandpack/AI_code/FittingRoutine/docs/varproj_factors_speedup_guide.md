# Speeding up `_varproj_factors`

A practical guide, ordered by expected payoff per unit of risk. It separates
changes that are **bit-identical** (same floating-point operations, same
order) from those that are **only mathematically equivalent** (they change
rounding at the ~1e-16 level and therefore break a bit-for-bit gate).

## 0. Context

Problem structure (from the project notes):

| Config | N (samples) | M (poles) | Rows with ridge |
|---|---|---|---|
| RSSE | 189 | 24-150 | N + M (up to 339) |
| nogap | 1001 | 24-70 | up to 1071 |
| ag3au3 | 451 | 24-100 | up to 551 |

- `A = diag(sqrt(sw)) * Phi`, real float64, entries in (0, 1].
- With ridge, `A` is `vstack([A, lambda*I])`, `Y` and `D` get `M` zero rows.
- All M columns change at every optimizer step, so there is no
  incremental/low-rank update between calls. The one-entry factor cache is the
  only cross-call reuse.
- Validation standard so far: bit-for-bit identical outputs.
- Normal equations / Cholesky on `A^T A` are already rejected (squares the
  condition number). Keep it that way.

Note that RSSE is **not** tall-skinny (N = 189 vs. M up to 150). Rules of
thumb derived for `N >> M` degrade there. Measure, don't assume.

## 1. Cost model (flops, rows `R = N (+ M)`, `c = n_ch`)

| Step | Approx. flops |
|---|---|
| `qr(A)` (geqrf) | 2RM^2 |
| Form Q (orgqr) | 2RM^2 |
| `Q.T @ D` (D has 2M columns) | 4RM^2 |
| `Q @ (Q.T @ D)` | 4RM^2 |
| `K = solve_triangular(R1, Q.T).T` | RM^2 |
| `Q.T @ Y`, `A @ X` | 2RMc each |
| `S = Z.T @ Res` | 4RMc |

So the Jacobian-only work (Z, K, S) is about `9RM^2 + 4RMc`, versus
about `4RM^2 + 4RMc` for the cost-only work (QR, Q, `Q.T@Y`, `A@X`). **Most of
the flops are only needed when the Jacobian is requested.** That drives the
top recommendation.

Run a profile on a realistic case first:

```python
# pip install line_profiler  (then: kernprof -l -v script.py)
@profile
def run():
    for _ in range(200):
        _varproj_factors(E, B, theta, sw, rcond, ridge)
run()
```

Also count how often the SVD fallback fires. It is far more expensive than the
fast path, and if it fires often it becomes the top item:

```python
_FALLBACK_COUNT = 0
# in the except branch:
global _FALLBACK_COUNT; _FALLBACK_COUNT += 1
```

---

## 2. Tier A: bit-identical changes

### A1. Split cost-only work from Jacobian work (largest, exact)

In a trust-region or LM loop, rejected trial steps need only the cost (`Res`,
`X`). Z, K and S are wasted there. If your optimizer evaluates the cost at
trial points and the Jacobian only at accepted ones (as `scipy.optimize.
least_squares` does), cache a lightweight core on the first call and compute
the Jacobian pieces lazily.

First check the accepted/trial ratio. The lower it is, the more this gains.

```python
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular

@dataclass
class _Core:
    theta: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    Q: np.ndarray | None
    R1: np.ndarray | None
    X: np.ndarray
    Res: np.ndarray
    ridge: float
    svd_path: bool

def _factors_core(E, B, theta, sample_weights, rcond, ridge):
    e, w = theta[0::2], theta[1::2]
    Phi = lorentzian_basis(E, e, w)
    A, Y = _weighted_design(Phi, B, sample_weights)
    M = e.size
    if ridge and ridge > 0.0:
        A = np.vstack([A, ridge * np.eye(M)])
        Y = np.vstack([Y, np.zeros((M, Y.shape[1]))])
    try:
        Q, R1 = np.linalg.qr(A)
        if not np.all(np.isfinite(R1)):
            raise np.linalg.LinAlgError
        d = np.abs(np.diag(R1))
        if d.min() <= 1e-12 * max(d.max(), 1e-300):
            raise np.linalg.LinAlgError
        X = np.linalg.solve(R1, Q.T @ Y)
        Res = A @ X
        Res -= Y
        return _Core(theta, A, Y, Q, R1, X, Res, ridge, svd_path=False)
    except np.linalg.LinAlgError:
        X, *_ = np.linalg.lstsq(A, Y, rcond=rcond)
        Res = A @ X
        Res -= Y
        return _Core(theta, A, Y, None, None, X, Res, ridge, svd_path=True)

def _build_D(E, theta, sample_weights, ridge):
    e, w = theta[0::2], theta[1::2]
    M = e.size
    dPhi_de, dPhi_dw = basis_derivatives(E, e, w)
    D = np.stack([dPhi_de, dPhi_dw], axis=2).reshape(dPhi_de.shape[0], 2 * M)
    if sample_weights is not None:
        sw = np.sqrt(np.asarray(sample_weights, dtype=float))
        np.multiply(D, sw[:, None], out=D)
    if ridge and ridge > 0.0:
        D = np.vstack([D, np.zeros((M, 2 * M))])
    return D

def _factors_jac(core, E, sample_weights, rcond):
    D = _build_D(E, core.theta, sample_weights, core.ridge)
    if not core.svd_path:
        Z = core.Q @ (core.Q.T @ D)
        np.subtract(D, Z, out=Z)
        K = solve_triangular(core.R1, core.Q.T).T
    else:
        A = core.A
        coef, *_ = np.linalg.lstsq(A, D, rcond=rcond)
        Z = A @ coef
        np.subtract(D, Z, out=Z)
        pinvI, *_ = np.linalg.lstsq(A, np.eye(A.shape[0]), rcond=rcond)
        K = pinvI.T
    return Z, K, Z.T @ core.Res
```

Every operation is the same as in the original, applied to the same inputs in
the same order, so the outputs are bit-identical. The cache entry now holds
`Q` and `R1`, so key it on `theta` as before. The basis derivatives are no
longer computed on cost-only calls.

### A2. Make K optional

`K` costs about `RM^2` and is only needed for the full Golub-Pereyra
Jacobian. If any caller uses the Kaufman approximation, or only needs the
gradient, add `need_K=True` and return `None` when it is off. The remaining
outputs are unchanged.

### A3. `check_finite=False` on the triangular solve

`solve_triangular(R1, Q.T)` scans the right-hand side for NaN/inf. `R1` is
already checked above, and `Q` comes from Householder reflections of a finite
`A`, so the scan is redundant:

```python
K = solve_triangular(R1, Q.T, check_finite=False).T
```

The numerics are unchanged. The saving is small but free.

### A4. Threading (environment, no code change to numerics)

All heavy operations are BLAS/LAPACK-bound, and the matrices are small
(hundreds of rows, tens to ~150 columns). At these sizes multi-threaded BLAS
often loses to single-threaded because of synchronization overhead, and
oversubscription on a shared node makes it worse.

- Compare 1 thread per process against `--cpus-per-task` threads.
- Prefer 1 thread per process and parallelize across independent runs.
- Set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`
  explicitly in the Slurm script.

```python
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
    bench()
```

Changing thread counts can alter reduction order inside BLAS. Treat it as
"likely bit-identical, verify with the gate".

### A5. Build padded arrays once (small)

With ridge, `np.vstack` re-copies `A`, `Y` and `D`. Allocate the padded
arrays at full size and fill the top block:

```python
R = N + M if ridge else N
D = np.zeros((R, 2 * M))
D[:N, 0::2] = dPhi_de
D[:N, 1::2] = dPhi_dw
```

This produces the same values with fewer copies. Allocation was measured at
about 1-3% of the function, so this is low priority. (A plain preallocated
`D` without ridge saves nothing over `np.stack(...).reshape`, which is one
copy either way. It only helps if `basis_derivatives` can write into `out=`.)

### A6. Unpadded ridge products (verify with the gate)

The ridge rows of `Y` and `D` are zeros. You can skip them in the products:

```python
n = N
Qt, Qb = Q[:n], Q[n:]
QtY = Qt.T @ Y0                 # Y0: the N data rows
QtD = Qt.T @ D0
Z_top = D0 - Qt @ QtD
Z_bot = -(Qb @ QtD)
Res_bot = Qb @ (X_side)         # Res ridge rows are A_bot @ X
```

Mathematically exact, and it removes up to ~45% of the rows in these
products for RSSE. In floating point, GEMM on differently shaped operands can
block and sum in a different order, so this may not be bit-identical. Run the
gate before adopting it.

---

## 3. Tier B: mathematically equivalent, not bit-identical

Adopt these only if you relax the gate to "allclose plus same optimizer
trajectory at convergence" (see section 5).

### B1. One economy SVD in the fallback

The fallback currently runs three `lstsq` calls (three SVDs) and builds an
`(N+M) x (N+M)` identity, which costs O(R^2) memory and O(R^2 M) time. One
economy SVD gives everything:

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
r = np.finfo(A.dtype).eps * max(A.shape) if rcond is None else rcond
keep = s > r * s[0]
U, s, Vt = U[:, keep], s[keep], Vt[keep]

UtY = U.T @ Y
X = Vt.T @ (UtY / s[:, None])
Res = U @ UtY
Res -= Y
Z = U @ (U.T @ D)
np.subtract(D, Z, out=Z)
K = (U / s) @ Vt                # (A^+)^T
```

The truncation rule matches `lstsq` (singular values at or below
`rcond * s_max` are dropped). The results agree to rounding, but
`np.linalg.svd` (gesdd) and `lstsq` (gelsd) are different LAPACK drivers, so
they are not bit-identical. This only matters if the fallback fires often
(count it, see section 1).

### B2. `S = D^T Res` instead of `Z^T Res`

`Res` lies in the range of `(I - P_A)`, so `P_A Res = 0` and
`Z^T Res = D^T (I - P_A) Res = D^T Res`. This does not need `Z`, which helps
anywhere only the gradient is needed. With ridge, `D^T Res = D0^T Res[:N]`
because the ridge rows of `D` are zero.

### B3. Triangular solve for `X`, `Res` from `Q`

```python
QtY = Q.T @ Y
X = solve_triangular(R1, QtY, check_finite=False)
Res = Q @ QtY
Res -= Y
```

This saves one pass over `A` (no `A @ X`) and uses a triangular solve instead
of an LU-based `np.linalg.solve`. It changes rounding.

---

## 4. Things I would not do (and corrections)

- **Householder reflectors without forming Q (`qr_multiply`, geqrf + ormqr).**
  Applying `Q` or `Q^T` through reflectors costs about `4RM` flops per
  right-hand-side column, versus `2RM` for GEMM with an explicit `Q`. Your
  workload applies `Q` to many columns (`2M` for `D`, twice, plus `M` for `K`):

  | Route | Flops for the D-related work |
  |---|---|
  | Explicit Q | about 10 R M^2 (form Q, `Q.T@D`, `Q@(...)`) |
  | Reflectors | about 16 R M^2 - 8 M^3 |

  That is roughly 1.3x (RSSE, M near N) to 1.6x (nogap, N >> M) more flops
  for the reflector route. Blocked `ormqr` is efficient, but I don't expect it
  to close that gap. Also, `scipy.linalg.qr_multiply` refactorizes on each call,
  so you would need LAPACK-level `geqrf` + repeated `ormqr` calls. A claimed
  20-40% saving looks like it assumes few right-hand-side columns. Treat it as
  a benchmark experiment at RSSE sizes (R = 339, M = 150), not as the plan.
  It also breaks bit-for-bit reproducibility.
- **Cholesky on the ridge system.** Normal equations, already rejected.
- **`Rinv` + GEMM for K.** I suggested this earlier, and it was backwards on
  flops: `trsm` is about `RM^2`, GEMM would be `2RM^2`. Keep the triangular
  solve. If you want to avoid the two transposes, call BLAS directly:
  `scipy.linalg.blas.dtrsm(1.0, R1, Q, side=1, lower=0, trans_a=1)` solves
  `X R^T = Q`, i.e. `K = Q R^{-T}`. Likely bit-identical to `dtrtrs`, but
  verify.
- **F-ordered inputs to avoid a hidden QR copy.** I suggested this earlier.
  `np.linalg.qr` copies into Fortran layout internally regardless, so this
  does not help there. `scipy.linalg.qr(A, mode='economic', overwrite_a=True,
  check_finite=False)` can avoid the copy for F-ordered input, but it links
  scipy's BLAS rather than numpy's, so results may differ at rounding level.
  With allocation measured at 1-3%, skip it.

---

## 5. Validation and benchmarking

### Gate tiers

```python
def gate_exact(ref, new):
    return all(np.array_equal(a, b) for a, b in zip(ref, new))   # Tier A

def gate_close(ref, new, rtol=1e-9, atol=1e-12):
    return all(np.allclose(a, b, rtol=rtol, atol=atol)
               for a, b in zip(ref, new))                       # Tier B
```

Run the exact gate on all three configs, with and without ridge, and with
`sample_weights` both `None` and set. For Tier B, also compare the final fit:
same iteration count, same accepted/rejected pattern, and final cost agreeing
to a tolerance you set in advance. Z and K amplify rounding when `A` is
near-rank-deficient, so do not expect tight `allclose` on the fallback path.
Compare those against the SVD reference at a looser tolerance.

### Benchmark template

```python
import time, numpy as np
from threadpoolctl import threadpool_limits

CONFIGS = {"RSSE":   dict(N=189,  M=150, n_ch=...),
           "nogap":  dict(N=1001, M=70,  n_ch=...),
           "ag3au3": dict(N=451,  M=100, n_ch=...)}   # fill in n_ch

def bench(fn, args, reps=50, warm=5):
    for _ in range(warm): fn(*args)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*args); t.append(time.perf_counter() - t0)
    return np.median(t)

for name, cfg in CONFIGS.items():
    args = make_inputs(**cfg, ridge=1e-3)          # your own factory
    for nthr in (1, 4):
        with threadpool_limits(limits=nthr):
            print(name, nthr, bench(_varproj_factors, args))
```

Always benchmark at the **largest M** per config. That is where the
`N >> M` assumptions are weakest.

---

## 6. Recommended order

1. Measure: profile, count SVD fallbacks, and count accepted vs. trial steps.
2. **A1** (lazy Jacobian split) plus **A3** (`check_finite=False`). Free and
   exact.
3. **A2** (`need_K`) if any caller can use it.
4. **A4** (threads). Often the biggest win at these matrix sizes.
5. **A6** (unpadded ridge products), gated by the exact check.
6. If the fallback fires often, **B1**. If you are willing to relax the gate,
   also B2/B3.
7. Reflector-based QR: benchmark only, expect it to lose.
