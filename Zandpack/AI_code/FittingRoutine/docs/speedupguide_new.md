# Next steps: `_varproj_*` / `fit_poles` optimization

Continuation of `varproj_factors_speedup_guide.md`. This document records where
things stand and what to do next, in order, with the decision rule for each
step.

## 1\. Status

Done and validated (bit-identical, Gate A 6/6, 8-case unit identity):

- **A1** `_varproj_factors` split into `_varproj_core` (QR, X, Res) and `
  _varproj_jac` (D, Z, K, S), with a shared one-entry cache in `fit_poles`.
- **A3** `check_finite=False` on the K triangular solve.
- Dense-branch `jac` no longer recomputes the QR block after `fun` at the same
  theta.

Measured:


|Item|Result|
|-|-|
|Calls to `_varproj_factors` in a production fit|10809, of which 5154 feed a Jacobian (52.3% cost-only)|
|SVD fallback|~30 per fit (~0.3% of calls)|
|Threads|`OMP_NUM_THREADS=1` is faster|
|RSSE end-to-end|25.0 s -> 23.0 s (single run), result identical (n=42, rms 0.00592872)|
|Cumulative since 10-08 baseline|~31 s -> 23 s|
|Dominant remaining cost|scipy's internal trust-region SVD|

Rejected, with measurements: A6 (unpadded ridge products; not bit-exact, worst
8e-13, 0.91-1.07x), B1/B2/B3 (unneeded or gate-relaxing), reflector QR for the
many-RHS Jacobian work (about 1.3-1.6x more flops).

Open: nogap end-to-end benchmark is shelved until a memory-growth issue is
understood.

## 2\. Step 1: profile what is left (before any new code)

Goal: know the fraction of the 23 s spent in each place. By Amdahl, no further
change to `_varproj_*` can save more than its share.

```
import cProfile, pstats
cProfile.run("run_rsse_fit()", "/tmp/rsse.prof")
p = pstats.Stats("/tmp/rsse.prof")
p.sort_stats("cumulative").print_stats(30)

```
Split the cumulative time into:

1.  `fun` (core: basis, QR, solve)
2.  `jac` (`_varproj_jac` \+ `_assemble_dense_j`)
3.  scipy `svd` / `lstsq` inside `trf`
4.  everything else (residual assembly, Python overhead, numba kernel calls)

**Decision rule**


|Where the time is|Next move|
|-|-|
|scipy SVD > ~50%|Step 4 (shrink J), then Step 5|
|`_varproj_core` large|Step 6 (reflector cost-only path)|
|`_varproj_jac` large|Step 3 cleanup, then check n_ch (Step 4)|
|"other" large|Profile line-by-line in `fit_poles` and `_assemble_dense_j`|

## 3\. Step 2: investigate the nogap memory growth (blocks the nogap benchmark)

First separate "caused by A1" from "pre-existing":

```
# peak RSS, one nogap fit each, same inputs
/usr/bin/time -v python run_nogap_fit.py          # current
/usr/bin/time -v python run_nogap_fit_pre_A1.py   # backups/AI_fit_2026-10-09_pre_A1_split.py

```
If only the current version grows, check the cache. It must hold exactly one
entry, replaced on each new theta. Typical culprits: a dict keyed per core, a
closure keeping the previous entry, scipy retaining `fun` outputs.

```
import tracemalloc
tracemalloc.start(10)
# in fun(): every ~50 calls
snap = tracemalloc.take_snapshot()
for s in snap.statistics("lineno")[:10]:
    print(s)

```
If both versions grow, the cause is elsewhere (per-iteration histories, stored
Jacobians, numba cache objects) and A1 is not involved.

## 4\. Step 3: small bit-identical cleanups

Apply only if you are already in the code. Gains are small.

1.  Drop `A` and `Y` from the core dict on the QR path. `_varproj_jac` reads
    only `Q`, `R1`, `Res`, `theta` there. `A` is needed on the SVD path only.
    Grep for other readers of `core["A"]` and `core["Y"]` first.
2.  Store `theta.copy()` in the core, because `_varproj_D` rebuilds `D` from it
    later.
3.  Build `D` in one pass into a preallocated `(N(+M), 2M)` array instead of `
    stack` \+ `reshape` \+ `vstack`:

```
D = np.zeros((R, 2 * M))
D[:N, 0::2] = dPhi_de
D[:N, 1::2] = dPhi_dw        # multiply by sw[:, None] with out= when weights are given



```
Gate: `np.array_equal` on all outputs, ridge and no ridge, weights and none,
including the SVD-fallback case.

## 5\. Step 4: shrink the Jacobian that goes into scipy (largest likely lever)

With `tr_solver='exact'`, `trf` takes an SVD of the full Jacobian every
iteration. Its cost grows with (residual length) x (2M)^2, and the residual
length scales with n_ch. Channel compression (numerical rank ~9-19 versus 729
channels for ag3Au3) attacks this directly.

- This changes the residual vector, so it will **not** be bit-identical. Use the
  relaxed gate (section 8).
- Check that the compression basis is fixed within one `fit_poles` call. If it
  is recomputed per iteration, the optimizer sees a moving objective.
- Re-run Step 1 afterward. The split between core, Jacobian and scipy SVD will
  have changed, which decides Step 6.

## 6\. Step 5: solver choices (only if scipy SVD is still dominant)

These change the optimizer trajectory, so use the relaxed gate:

- `x_scale="jac"` often reduces iterations on badly scaled pole parameters
  (positions e_i versus widths w_i). Fewer iterations is a free speedup if the
  final cost is unchanged.
- `tr_solver="lsmr"` avoids the SVD but converges differently. Worth a test only
  for large J (many channels, M near 150).
- `method="lm"` (MINPACK) uses QR instead of SVD, but it does not support
  bounds. Only viable if the fit has no bounds on e_i, w_i.
- Tighten or loosen `ftol` / `xtol` / `gtol` only after checking how many final
  iterations improve the cost at the 1e-6 relative level. Do not change
  tolerances to gain speed on a fit used for reported results.

Change one thing at a time and record iteration count, `nfev`, `njev`, final
cost and wall time.

## 7\. Step 6: cost-only QR without forming Q (conditional)

After channel compression the cost-only right-hand side is `Y` with only n_ch
columns, which reverses the earlier flop comparison for that path. On rejected
steps you only need X and Res, not Q:

- `scipy.linalg.qr(A, mode="raw")` (geqrf) gives R and the reflectors.
- Apply `Q^T` to `Y` with ormqr. Skipping `orgqr` saves about 2RM^2 of the
  cost-only work.
- Form Q explicitly only when `_varproj_jac` is called (after an accepted step).

This breaks bit-for-bit and adds code, so benchmark it as a standalone variant
first, as with A6. Adopt it only if the cost-only path is a visible share of
the Step 1 profile and the standalone speedup exceeds ~10% on the cost-only
function.

## 8\. Validation standard going forward


|Change type|Gate|
|-|-|
|Same operations, same order (Step 3)|`np.array_equal` on all outputs plus identical `nfev`, `njev`, final cost|
|Changes rounding or op order (Step 6)|`np.allclose` per call, plus same iteration count and accepted/rejected pattern, final cost within a tolerance fixed in advance|
|Changes the problem or solver (Steps 4, 5)|Same final cost within tolerance, fitted poles within tolerance, and downstream transmission T = Tr\[G Gamma_R G^dagger Gamma_L] compared against the uncompressed fit|

Timing protocol: 3-5 alternating A/B runs, `OMP_NUM_THREADS=1`, report medians.
A single 25.0 s versus 23.0 s pair is not enough evidence for an ~8% effect.

## 9\. Housekeeping

- Update the original speed-up guide with the measured outcomes (adopted,
  rejected, the 52.3% share, A6 numbers) so it reflects what was done.
- Keep each backup file paired with the change that followed it (
  `backups/AI_fit_2026-10-09_pre_A1_split.py` and later).
- Record the stale numba cache issue: `cache=True` modules fail to load under
  dynamic names, so delete stale entries in `backups/__pycache__` when a backup
  is moved or renamed.

## 10\. Order of work

1.  Profile the 23 s RSSE fit (Step 1).
2.  Nogap memory growth: A1 versus pre-A1 peak RSS (Step 2).
3.  Channel compression with the relaxed gate (Step 4), then re-profile.
4.  `x_scale` and solver experiments if scipy SVD is still dominant (Step 5).
5.  Cost-only reflector path if the profile still shows the core as significant
    (Step 6).
6.  Small cleanups (Step 3) whenever the code is open anyway.
