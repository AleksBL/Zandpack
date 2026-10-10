# `tr_solver='lsmr'` vs `'exact'`: what the edge-band figure shows

Follow-up to `varproj_next_steps.md`, Step 5. Based on the VerySmallRSSE figure
(zoom [3,5], residuals, E=0 notch) and the measured timings.

## 1. Summary

- The lsmr result in [3,5] eV does **not ring**. It is smooth, with a positive
  offset relative to the reference. The exact solver is the one oscillating.
- The gate number (rms 0.01552 for lsmr vs 0.002904 for exact) is the rms of
  `T_fit - T_ref`. For lsmr it is dominated by that smooth offset, not by
  oscillation. Calling it "ringing" is a mislabel.
- The offset is still a real error: about 0.008-0.010 (roughly 1.5% of T)
  mid-band, growing to about 0.034 at the 5 eV window edge.
- The right decision is per system and depends on which observables matter
  (section 4). The single "ringing" gate should be split into a bias metric and
  a ringing metric (section 6).

## 2. What the figure shows

| Quantity | exact | lsmr |
|---|---|---|
| Residual in [3,5] | oscillates, about +/-0.005 (max about 0.008), roughly zero mean | smooth, same sign, 0.008-0.010 rising to about 0.034 at 5 eV |
| rms vs reference in [3,5] | 0.002904 | 0.01552 |
| T(0), E=0 notch (ref 0.0224) | 0.0240 | 0.0214 |
| Pole counts shown in legend | 0: 50, 1: 75 | 0: 67, 1: 90 |

Elsewhere in the window (full-window panel, [-1,1] zoom) the two curves are
visually indistinguishable from the reference.

## 3. Why lsmr behaves differently (hypotheses, not verified)

As far as I recall the scipy documentation, with `tr_solver='lsmr'` the `trf`
method does not solve the trust-region subproblem exactly. It takes an inexact
Gauss-Newton step from lsmr and solves the subproblem in a two-dimensional
subspace spanned by that step and the gradient. The exact solver uses the full
SVD of J, including every small singular direction.

Consequences for a shared-pole Lorentzian fit:

1. **Implicit regularization.** Pole positions and widths are strongly coupled,
   and J is ill-conditioned when poles approach each other. lsmr at its default
   tolerances effectively filters weak singular directions (early stopping),
   which suppresses oscillation but can leave a smooth under-fit.
2. **Different basin.** The two solvers follow different trajectories on a
   non-convex problem, so they end in different local solutions. Existing
   tuning (ridge, pole initialization, adaptive pole-count loop, tolerances)
   was done around the exact solver's trajectory.
3. **Inexact step.** Loose `atol`/`btol` change trust-region radius updates and
   the `ftol`/`xtol`/`gtol` stopping behavior.

My earlier suggestion that the lsmr solution had worse between-sample behavior
(near-coincident poles, spikes) was wrong in the specifics: the figure shows
smoothing, not extra oscillation.

## 4. Which error costs what

| Observable | Favors | Why |
|---|---|---|
| Integrals, e.g. current from `integral T(E)[f_L - f_R] dE`, if the bias window reaches [3,5] | exact | a zero-mean oscillation averages out, a same-sign offset accumulates |
| Derivatives, dT/dE, NDR features | lsmr | oscillation creates spurious slope |
| E=0 notch depth | lsmr | T(0) closer to the reference (error about -0.0010 vs +0.0016) |
| Features far from E_F | either | the offset is likely irrelevant there |

## 5. Caveats on the comparison

- **Pole counts differ.** lsmr's adaptive loop ended with about 15 more poles
  per group (67/90 vs 50/75). The speed ratio (1.55x RSSE, 1.76x nogap) and the
  quality comparison are therefore not like-for-like. A smooth bias despite more
  poles points to optimization or stopping behavior, not model capacity.
- **Gamma versus T.** lsmr gave an 11x better Gamma fit overall, yet T in [3,5]
  is biased. Inspect the Gamma residual in [3,5] for both solvers and the
  `sample_weights` there. A small Gamma error near a steep band edge can map to
  a much larger T error. The reference falls off sharply just past 5 eV, so a
  bias growing toward the window edge is typical of a truncated Lorentzian fit.
- **Crossover.** lsmr was 1.56x slower below M of about 40. A dense SVD costs
  O(m n^2) in BLAS-3 form, lsmr costs O(k m n) with memory-bound matrix-vector
  products, and it only wins when k is much smaller than n. At n = 2M of about 80
  there is no room for that.
- One figure is one system and one start. Treat the conclusions as provisional
  until the checks in section 7 are done.

## 6. Replace the single "ringing" gate with two metrics

Split the residual in [3,5] into a low-pass part (bias) and a high-pass part
(ringing), and optionally add an integrated check.

```python
import numpy as np
from scipy.signal import savgol_filter

def split_residual(E, T_fit, T_ref, lo=3.0, hi=5.0,
                   smooth_width_eV=0.5, polyorder=3):
    """Assumes a uniform dense grid E. smooth_width_eV should span a few
    oscillation periods, so the smoother follows the bias but not the ripple."""
    m = (E >= lo) & (E <= hi)
    Ew, r = E[m], (T_fit - T_ref)[m]

    dE = np.median(np.diff(Ew))
    win = int(round(smooth_width_eV / dE))
    win += (win % 2 == 0)                      # must be odd
    win = min(win, len(r) if len(r) % 2 else len(r) - 1)

    smooth = savgol_filter(r, win, polyorder)  # low-pass: the bias
    ripple = r - smooth                        # high-pass: the oscillation

    integ = lambda y: np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(Ew))
    return dict(
        bias_mean=smooth.mean(),
        bias_max=np.abs(smooth).max(),
        ringing_rms=np.sqrt(np.mean(ripple ** 2)),
        int_rel_err=integ(T_fit[m] - T_ref[m]) / integ(T_ref[m]),
    )
```

Set tolerances for `bias_max`, `ringing_rms` and `int_rel_err` from what the
downstream quantity can tolerate, and record them with the gate. By eye, lsmr
would pass `ringing_rms` easily and be judged on bias alone.

## 7. Checks to run

Cheapest first:

1. **Train versus dense-grid error.** Evaluate both solutions on the fitting
   grid and on a much denser grid, especially in [3,5]. This shows whether the
   lsmr offset is also present at the fit points or only between them.
2. **Equal pole count.** Force both solvers to the same M (disable or cap the
   adaptive growth) and compare again. Removes the 50/75 vs 67/90 confound.
3. **Tighten lsmr.** Run with
   `tr_options={'atol': 1e-10, 'btol': 1e-10}` and a larger `maxiter`. If the
   result moves toward the exact solver's, inexactness matters. If not, it is
   the 2-D subspace restriction or a different basin.
4. **Warm start across solvers.** Start lsmr from the exact solution and the
   exact solver from the lsmr solution. If each stays in its own basin, the
   problem has several nearby optima.
5. **Gamma residual and weights in [3,5].** Compare the Gamma residuals of
   both solvers there and inspect `sample_weights` (see section 5).
6. **Compare solution structure.** Minimum pole separation, widths relative to
   the sample spacing, `||X||`, and the condition number of `A` at the solution.
7. **Basin variance.** Run the exact solver from perturbed initial poles and
   record the spread of `bias_max` and `ringing_rms`. If lsmr's values fall
   inside that spread, a single-run comparison cannot separate the solvers.
8. **Ridge.** A modest increase of `ridge` for the lsmr runs, to see whether the
   edge-band bias responds.

## 8. Current state and decision

- `tr_solver` knob is in `fit()` / `fit_poles`. Default `None` (exact) stays
  bit-identical and gated.
- lsmr is available as a per-system opt-in: `fit(..., tr_solver='lsmr')`.
- Reasonable rule of thumb, pending section 7: use lsmr only for large M
  (above roughly 40), where it is 1.5-1.8x faster, and only for systems where
  derivative-type quantities or the notch matter more than the smooth offset in
  the band edge. Keep exact for results that feed integrated quantities across
  the full window.
- After section 6 is in place, re-run the lsmr comparison on the other
  systems before deciding any opt-in.
