"""AI_fit_base - shared-pole Lorentzian fit of Hermitian matrix
functions + adaptive level-width sampling refinement, single-file
edition (merged from the lorentz_fit package + the historical
refine_sampling.py;
see the project HANDOFF for the full documentation and the migration
plan for the Zandpack context).

QUICK START (replaces the PoleGuess -> R.Fit -> curvefit_all ->
iterative_PSD -> FitNO2O block of a 3.py script):

    fits = {}
    for lead in range(R.num_leads):
        for ik in range(len(R.Nonortho_Gammas[0].vals[0])):
            E, F = extract_dense(R, lead, ik)
            fits[(lead, ik)] = fit_boosted(
                E, F,
                dict(n_init=24, max_poles=100, target_rms=2e-3,
                     psd=True, psd_force=True,
                     eta_raw=1e-2,        # contour imaginary part
                     eta_floor_mult=5.0,  # per-system! see table below
                     w_max_raw=0.5, local_floor_frac=1.5,
                     channel_tol=1e-4 if F.shape[1] > 32 else 0.0),
                n_res=8)
    inject_fits(R, fits)          # setLval + FitNO2O internally
    C = find_correction(R, Emin=-3, Emax=3.0)   # your existing tail
    R.Renormalise_H(C)

MULTI-K-POINT SYSTEMS (n_ik > 1): nothing extra to do - inject_fits
guarantees ONE common pole count for every (lead, ik) set (Zandpack's
single R.NumL requires it): smaller sets are padded to the global max
with inert zero-residue duplicates of pole 0. To equalize the fits
THEMSELVES (e.g. before pickling): equalize_pole_counts(fits,
mode='dup'|'zero'). See pad_poles/equalize_pole_counts docstrings.

TUNED KNOBS PER SYSTEM (measured; eta_floor_mult is the sharp one):
    zgnr            mult 5.0  w_max 0.35 psd_force True
    zgnr_spin       mult 5.0  w_max 0.5  psd_force True  (0.35 regresses)
    VerySmallRSSE   mult 3.0  w_max 0.5  psd_force False (E=0 notch)
    agnr            mult 1.0  w_max 0.5  psd_force False (mult 5 kills it)
    spinpol_ag5     mult 2.0  w_max 0.5  psd_force False
    Fig2Ag5         mult 1.0  w_max 0.5  psd_force False
    Ag3Au3(-gap)    mult 5.0  w_max 0.5  psd_force False
    AuBreakJunction mult 2.0  w_max 0.5  psd_force False
Keep OMP_NUM_THREADS=1 (measured faster AND reproducible).

MATHEMATICS (for anyone picking this up; HANDOFF 4 has the full story):

MODEL. F(E) = sum_i C_i * w_i^2 / ((E - e_i)^2 + w_i^2), with C_i
Hermitian p x p residues SHARED across all matrix entries (one pole
set for the whole matrix function). Strictly proper (decays 1/E^2:
no constant/linear/dispersive terms). Widths w_i are real, so each
term is its own Kramers-Kronig partner (the pole pair e_i +- i w_i):
the assembled Gamma is the absorptive part of a causal response.

SCALING (prepare/FitData). E' = (E - mid)/scale, F' = F/f_scale —
pure conditioning; every output is unscaled exactly.

CHANNEL VECTORIZATION. A Hermitian p x p matrix is a real vector of
p^2 channels: p real diagonals + 2 * p(p-1)/2 (re, im) off-diagonals
(hermitian_channels/pack_responses give B in (n_ch, N)). At fixed
poles the LS cost is SEPARABLE per channel, so the linear stage is
one real LS with n_ch right-hand sides.

VARIABLE PROJECTION. At fixed poles theta = (e_1, w_1, ..., e_M, w_M)
the residues solve by weighted LS — ridge-augmented design
[A; lambda I] (rows M) so the outer cost is exactly
||A X - Y||^2 + lambda^2 ||X||^2 — solved via thin QR of the padded
design (_varproj_core; rank-deficient pole sets fall back to a
rank-truncated SVD path). The outer problem then optimizes ONLY the
2M pole parameters (not 2M + M p^2), using scipy least_squares (trf,
bounds) on the Golub-Pereyra Jacobian
    dR/dtheta_j = (I - P_A) dA_j beta - A (A^T A)^{-1} dA_j^T R
(the second term drops from the cost gradient — Kaufman). A1: the
core (QR, X, Res) and the Jacobian pieces (D, Z, K, S) are split so
cost-only trial steps skip the Jacobian work.

PSD MACHINERY. Physical Gamma(E) >= 0 everywhere (including BETWEEN
samples) forces every C_i PSD. project_psd = per-pole eigenvalue
clip; psd_solve_admm solves the CONVEX PSD-constrained LS at fixed
poles exactly (ADMM: X-step = one (M x M) cholesky solve with
proximal ridge toward the projected variable, Z-step = PSD
projection, over-relaxation alpha = 1.6); psd_refine warm-starts
from clip+scale and keeps the better. psd_dict_init: NNLS
non-negative dictionary fit of Tr Gamma — positivity-native poles.

CHANNEL COMPRESSION. B is numerically LOW RANK for smooth
level-width data (rank 9-19 of 729 channels measured on real Ag3Au3).
The LS cost is invariant under orthogonal rotation of the channel
axis, so the OPTIMIZER runs on the r strongest singular directions
(compress_channels, fixed within one fit) while the FINAL coefficient
solve always uses the full responses (exact X/R).

WIDTH FLOORS / BOUNDS (oscillation control). Poles narrower than
the sample spacing (or the data's eta broadening) are invisible to
the samples — their LS columns are nearly parallel and they only
produce between-sample/off-window artifacts; w_min floors forbid
them (eta_raw/eta_floor_mult, local_floor_frac near sparse samples).
Very broad poles (w ~ window span) act as an unconstrained constant
background with a sweeping KK ramp; w_max_raw forbids those.

WEIGHTS. 'relative': per-sample 1/max(||F_n||, eps) (half-relative —
deliberate; fully-relative degrades peak fidelity), plus trapezoid
quadrature weights so refined (non-uniform) grids do not over-weight
dense clusters.

ADAPTIVE LOOP (_fit_impl). init (skrf relocation / NNLS dictionary /
linspace) -> prune -> residual-peak pole insertions -> PSD stages
with ACCEPT-GUARDS (candidates ranked by PSD-projected rms) -> local
width-floor stage -> two-stage w_min (tight, then loose only if the
tight fit is inadequate) -> polish. fit_boosted adds a SECOND small
PSD-forced fit of the residual (CombinedFit; the sum stays PSD).

SOLVER (tr_solver). 'exact': dense SVD of J per trust-region step —
error is a near-zero-mean ripple that averages out in integrals.
'lsmr' (DEFAULT): inexact steps, 1.5-1.8x faster at M >~ 40 — error
at steep band edges is a same-sign BIAS that accumulates in
integrals. 'gram': vendored trf loop that never forms J — only
G = J^T J and g = J^T f from the varproj factors plus one eigh per
step (never_form_J guide); follows the 'exact' trajectory, weak
directions below gram_cut are truncated. Per-method caveats +
measurements: HANDOFF 4.11.

INERTNESS. Zero-residue poles contribute nothing to the model, so
pole-count padding (pad_poles/equalize_pole_counts/inject_fits) is
exact: evaluate() is unchanged, NaN-safety via real-width
duplicates ('dup') or far-placed zero-width poles ('zero').

Sections below mirror the original modules (data, varproj, poles_init,
prune, constraints, validate, sampling, pipeline, TD glue).
"""
import gzip
import lzma
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares, nnls, OptimizeResult
from scipy.signal import find_peaks
from scipy.sparse.linalg import LinearOperator

try:  # optional: ~2-4x faster dense Jacobian assembly (see below)
    from numba import njit as _njit
    from numba import config as _numba_config
    if _numba_config.DISABLE_JIT:
        # NUMBA_DISABLE_JIT=1 turns the kernel into pure-Python loops
        # (slower than the numpy path) - fall back to numpy instead
        _njit = None
except ImportError:  # pragma: no cover - numpy fallback stays exact
    _njit = None

# --- environment invariants (measured, HANDOFF 4.11) -----------------------
# OMP_NUM_THREADS=1 is REQUIRED: threaded BLAS is ~11% SLOWER here
# post-compression AND breaks run-to-run reproducibility (threaded
# reduction order flips trust-region trajectories on knife-edge
# basins). Warn once at import if it is not set - the caller can
# still fix it before the first fit (the threads are read lazily by
# BLAS at the first large operation).
_omp = os.environ.get("OMP_NUM_THREADS", "")
if _omp != "1":
    warnings.warn(
        f"OMP_NUM_THREADS is {_omp!r}, not '1': threaded BLAS is "
        "measured SLOWER for this workload and BREAKS run-to-run "
        "reproducibility (different fit basins). Set "
        "OMP_NUM_THREADS=1 in the environment (before numpy is "
        "imported) or expect nondeterministic results.",
        RuntimeWarning, stacklevel=2)

__all__ = ["fit", "fit_fixed_poles", "FitResult", "prepare", "FitData", "fit_boosted",
           "CombinedFit", "inject_fits", "extract_dense", "resample",
           "refine_loop", "sampled_transmission_dos", "load_pickle",
           "current_lines", "suggest_insertions", "suggest_removals",
           "suggest_insertions_derivative", "validate",
           "suggest_pole_insertions"]


# ==========================================================================
# SECTION: data
# ==========================================================================
"""Data preparation for the shared-pole Lorentzian fit.

Holds samples F(E_n) in C^{p x p}, F(E_n) = F(E_n)^*, and rescales energies
and amplitudes to O(1) before fitting (the single biggest conditioning win).
All fitting happens in scaled units; the FitData helpers convert back.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FitData:
    """Scaled, Hermitized samples of a matrix-valued function.

    Attributes
    ----------
    E : (N,) real array, scaled energies in [-1, 1]
    F : (N, p, p) complex array, scaled Hermitian samples
    e_mid, e_scale : raw -> scaled mapping E' = (E - e_mid) / e_scale
    f_scale : raw -> scaled mapping F' = F / f_scale
    hermiticity_violation : max |F - F^H| before symmetrization (raw units)
    """

    E: np.ndarray
    F: np.ndarray
    e_mid: float
    e_scale: float
    f_scale: float
    hermiticity_violation: float

    def unscale_centers(self, e: np.ndarray) -> np.ndarray:
        """Scaled pole centers -> raw energies."""
        return np.asarray(e) * self.e_scale + self.e_mid

    def scale_centers(self, e_raw: np.ndarray) -> np.ndarray:
        """Raw energies -> scaled pole centers."""
        return (np.asarray(e_raw) - self.e_mid) / self.e_scale

    def unscale_widths(self, w: np.ndarray) -> np.ndarray:
        """Scaled widths -> raw units (same units as energy)."""
        return np.asarray(w) * self.e_scale

    def unscale_coeffs(self, C: np.ndarray) -> np.ndarray:
        """Scaled residue matrices -> raw units."""
        return np.asarray(C) * self.f_scale

    def evaluate_scaled(self, e, w, C, E_scaled):
        """Model evaluated on scaled energies, scaled units."""

        Phi = lorentzian_basis(E_scaled, e, w)
        return basis_apply(Phi, C)

    def evaluate(self, e_scaled, w_scaled, C_scaled, E_raw):
        """Model evaluated on raw energies, raw units."""
        E_scaled = (np.asarray(E_raw, dtype=float) - self.e_mid) / self.e_scale
        F_scaled = self.evaluate_scaled(e_scaled, w_scaled, C_scaled, E_scaled)
        return F_scaled * self.f_scale


def prepare(E, F, symmetrize: bool = True) -> FitData:
    """Validate, Hermitize and scale samples of a Hermitian matrix function.

    Parameters
    ----------
    E : (N,) array-like, real energies (any unit)
    F : (N, p, p) array-like, complex samples
    symmetrize : replace F by (F + F^H)/2 (exact Hermiticity; real diagonal)
    """
    E = np.asarray(E, dtype=float).ravel()
    F = np.asarray(F, dtype=complex)
    if F.ndim != 3 or F.shape[1] != F.shape[2]:
        raise ValueError(f"F must have shape (N, p, p), got {F.shape}")
    if F.shape[0] != E.size:
        raise ValueError(f"E has {E.size} samples but F has {F.shape[0]}")
    if E.size < 2:
        raise ValueError("need at least 2 samples")
    if not np.all(np.isfinite(F)):
        raise ValueError("F contains non-finite values")

    herm_viol = float(np.max(np.abs(F - np.conj(np.swapaxes(F, -1, -2)))))
    if symmetrize:
        F = 0.5 * (F + np.conj(np.swapaxes(F, -1, -2)))
    elif herm_viol > 1e-8 * max(1.0, float(np.max(np.abs(F)))):
        raise ValueError(f"F is not Hermitian (violation {herm_viol:.3e}); "
                         "pass symmetrize=True")

    e_mid = 0.5 * (E.min() + E.max())
    e_scale = 0.5 * (E.max() - E.min())
    if e_scale <= 0:
        e_scale = 1.0
    f_scale = float(np.max(np.linalg.norm(F, axis=(1, 2))))
    if f_scale <= 0:
        raise ValueError("F is identically zero")

    return FitData(
        E=(E - e_mid) / e_scale,
        F=F / f_scale,
        e_mid=float(e_mid),
        e_scale=float(e_scale),
        f_scale=f_scale,
        hermiticity_violation=herm_viol,
    )


# ==========================================================================
# SECTION: varproj
# ==========================================================================
"""Variable projection: shared-pole Lorentzian basis, inner linear LS,
and the outer nonlinear fit over pole positions.

Model (scaled units):

    F(E) = sum_i C_i * w_i^2 / ((E - e_i)^2 + w_i^2),   C_i = C_i^*

The basis Phi[n, i] is real, so real and imaginary parts of the Hermitian
upper triangle decouple into independent real linear channels (the lower
triangle is never fitted redundantly).
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator


def lorentzian_basis(E, centers, widths):
    """Phi[n, i] = w_i^2 / ((E_n - e_i)^2 + w_i^2). Real, shape (N, M)."""
    E = np.asarray(E, dtype=float)[:, None]
    d = E - np.asarray(centers, dtype=float)[None, :]
    w2 = np.square(np.asarray(widths, dtype=float))[None, :]
    return w2 / (d * d + w2)


def hermitian_channels(p: int):
    """Channels covering a Hermitian p x p matrix.

    Returns a list of (a, b, part) with part in {'re', 'im'}; off-diagonal
    entries contribute two real channels, diagonal entries one. The number
    of channels is p * (p + 1) / 2 * ... = p (diag) + 2 * p(p-1)/2.
    """
    chans = []
    for a in range(p):
        chans.append((a, a, "re"))
        for b in range(a + 1, p):
            chans.append((a, b, "re"))
            chans.append((a, b, "im"))
    return chans


def pack_responses(F, chans):
    """B[c, n] = real or imaginary part of F[n] along channel c. (n_ch, N)

    The channel packing is the real-vector isomorphism of the
    Hermitian matrix: F[n] <-> B[:, n] (p^2 real channels), so the
    Frobenius cost ||F_fit - F||_F^2 equals the plain 2-norm cost
    ||B_fit - B||^2 (off-diagonals appear once as re+im channels,
    not twice - the conjugate is not repeated).

    Vectorized: the channel loop runs n_ch times per call and sits on
    hot paths (every candidate evaluation in the adaptive loop)."""
    F = np.asarray(F)
    n_ch = len(chans)
    B = np.empty((n_ch, F.shape[0]))
    if n_ch == 0:
        return B
    ia = np.fromiter((c[0] for c in chans), dtype=int, count=n_ch)
    ib = np.fromiter((c[1] for c in chans), dtype=int, count=n_ch)
    re = np.fromiter((c[2] == "re" for c in chans), dtype=bool,
                     count=n_ch)
    vals = F[:, ia, ib]                      # (N, n_ch)
    B = np.where(re[None, :], vals.real, vals.imag).T
    return B


def channel_indices(chans):
    """Precomputed index arrays for a Hermitian channel list:
    (ia, ib, re_mask). Shared by the vectorized pack/unpack paths."""
    n_ch = len(chans)
    ia = np.fromiter((c[0] for c in chans), dtype=int, count=n_ch)
    ib = np.fromiter((c[1] for c in chans), dtype=int, count=n_ch)
    re = np.fromiter((c[2] == "re" for c in chans), dtype=bool,
                     count=n_ch)
    return ia, ib, re


def unpack_coefficients(X, chans, p):
    """X: (M, n_ch) real -> C: (M, p, p) complex Hermitian (vectorized)."""
    M = X.shape[0]
    C = np.zeros((M, p, p), dtype=complex)
    if len(chans) == 0:
        return C
    ia, ib, re = channel_indices(chans)
    d = re & (ia == ib)
    o = re & ~d
    im = ~re
    C[:, ia[d], ib[d]] = X[:, d]
    C[:, ia[o], ib[o]] += X[:, o]
    C[:, ib[o], ia[o]] += X[:, o]
    C[:, ia[im], ib[im]] += 1j * X[:, im]
    C[:, ib[im], ia[im]] -= 1j * X[:, im]
    return C


def basis_derivatives(E, centers, widths):
    """dPhi/de and dPhi/dw for the Lorentzian basis.

    Phi = w^2 / D, D = (E-e)^2 + w^2
    dPhi/de = 2 w^2 (E-e) / D^2
    dPhi/dw = 2 w (E-e)^2 / D^2
    Returns (dPhi_de, dPhi_dw), each (N, M) real.
    """
    E = np.asarray(E, dtype=float)[:, None]
    e = np.asarray(centers, dtype=float)[None, :]
    w = np.asarray(widths, dtype=float)[None, :]
    d = E - e
    D = d * d + w * w
    D2 = D * D
    return (2.0 * w * w * d / D2, 2.0 * w * d * d / D2)


def _weighted_design(Phi, B, sample_weights):
    """Return (A, Y): weighted design matrix and responses.

    A = diag(sqrt(sw)) @ Phi and Y = diag(sqrt(sw)) @ B^T, so any
    later unweighted product A^T A / A^T Y equals the WEIGHTED normal
    equations sum_n sw_n Phi_n Phi_n^T (etc.): the sample weights
    enter exactly once, here, through a real diagonal scaling."""
    if sample_weights is None:
        return np.asarray(Phi), np.asarray(B).T
    sw = np.sqrt(np.asarray(sample_weights, dtype=float))
    return (np.asarray(Phi) * sw[:, None],
            np.asarray(B).T * sw[:, None])


def solve_coefficients(Phi, B, sample_weights=None, rcond=None, ridge=0.0):
    """Weighted linear LS for all Hermitian channels at fixed poles.

    Phi : (N, M) real basis, B : (n_ch, N) real responses.
    sample_weights : optional (N,) non-negative weights (e.g. 1/||F_n||).
    ridge : optional Tikhonov penalty lambda on the coefficients
        (min ||A X - Y||^2 + lambda^2 ||X||^2), implemented exactly by
        augmenting the design with lambda * I rows. It only bites on
        near-null directions of the sample LS - coincident poles,
        sub-resolution poles, or multiple window-spanning broad poles,
        whose unpenalized coefficients are astronomically cancelling
        (the between-samples / off-window spike mechanism).

    Returns (X, R): coefficients X (M, n_ch) and weighted residuals R (n_ch, N).
    One SVD-based lstsq handles all channels jointly.
    """
    A, Y = _weighted_design(Phi, B, sample_weights)
    if ridge and ridge > 0.0:
        A = np.vstack([A, ridge * np.eye(A.shape[1])])
        Y = np.vstack([Y, np.zeros((A.shape[1], Y.shape[1]))])
    X, _, _, _ = np.linalg.lstsq(A, Y, rcond=rcond)
    R = (A @ X - Y)[:Phi.shape[0]].T if ridge and ridge > 0.0 else (A @ X - Y).T
    return X, R


def _varproj_core(E, B, theta, sample_weights, rcond, ridge=0.0):
    """Cost-only part of the variable projection (guide A1): basis,
    weighted design, QR, coefficients X and residual Res. The Jacobian
    pieces (D, Z, K, S) live in _varproj_jac and are computed lazily -
    scipy's trial steps only need this core. Same operations, same
    order as the combined computation, so bit-identical."""
    e = theta[0::2]
    w = theta[1::2]
    Phi = lorentzian_basis(E, e, w)
    A, Y = _weighted_design(Phi, B, sample_weights)
    M = e.size
    if ridge and ridge > 0.0:
        A = np.vstack([A, ridge * np.eye(M)])
        Y = np.vstack([Y, np.zeros((M, Y.shape[1]))])
    try:
        Q, R1 = np.linalg.qr(A)                    # thin QR, no pivoting
        if not np.all(np.isfinite(R1)):
            raise np.linalg.LinAlgError
        # nearly-coincident poles make A rank-deficient; the triangular
        # solve would then silently return astronomically cancelling
        # coefficients (the between-samples/off-window spikes). Detect and
        # fall back to the rank-truncated, minimum-norm SVD path.
        d = np.abs(np.diag(R1))
        if d.min() <= 1e-12 * max(d.max(), 1e-300):
            raise np.linalg.LinAlgError
        X = np.linalg.solve(R1, Q.T @ Y)           # (M, n_ch)
        Res = A @ X                                # (N+M, n_ch), in place:
        Res -= Y
        # guide step 3: the QR path never needs A/Y again (only the SVD
        # fallback reads core["A"]); theta is copied defensively since
        # _varproj_D rebuilds D from it later
        return dict(theta=theta.copy(), Q=Q, R1=R1, X=X, Res=Res,
                    ridge=ridge, svd_path=False)
    except np.linalg.LinAlgError:
        X, _, _, _ = np.linalg.lstsq(A, Y, rcond=rcond)
        Res = A @ X
        Res -= Y
        return dict(theta=theta.copy(), A=A, Q=None, R1=None, X=X, Res=Res,
                    ridge=ridge, svd_path=True)


def _varproj_D(E, theta, sample_weights, ridge):
    """Weighted, ridge-padded derivative design (N(+M), 2M),
    interleaved layout [de_0, dw_0, de_1, dw_1, ...] matching theta.
    Built in one pass into a preallocated array (guide step 3)."""
    e = theta[0::2]
    w = theta[1::2]
    M = e.size
    dPhi_de, dPhi_dw = basis_derivatives(E, e, w)
    N = dPhi_de.shape[0]
    R = N + M if (ridge and ridge > 0.0) else N
    D = np.zeros((R, 2 * M))
    D[:N, 0::2] = dPhi_de
    D[:N, 1::2] = dPhi_dw
    if sample_weights is not None:
        sw = np.sqrt(np.asarray(sample_weights, dtype=float))
        np.multiply(D[:N], sw[:, None], out=D[:N])
    return D


def _varproj_jac(core, E, sample_weights, rcond, with_k=True):
    """Jacobian-only pieces (Z, K, S) from a cached core (guide A1).
    Bit-identical to the combined computation.

    with_k=False skips the K triangular solve on the QR path (the Gram
    solver never needs K there - never_form_J guide section 7 item 2;
    K=None is returned instead). The SVD fallback always returns K: the
    Gram factors need it (R1/Q do not exist there)."""
    D = _varproj_D(E, core["theta"], sample_weights, core["ridge"])
    if not core["svd_path"]:
        Q, R1 = core["Q"], core["R1"]
        # Z = D - Q (Q^T D): one matmul buffer, subtract in place (D is
        # not used again in this branch)
        Z = Q @ (Q.T @ D)
        np.subtract(D, Z, out=Z)
        # K = A (A^T A)^{-1} = Q R^{-T} = (R^{-1} Q^T)^T
        # (guide 4.8: R1 is upper triangular - triangular solve;
        # A3: R1 is finite (checked) and Q is Householder of a finite
        # A, so the RHS NaN/inf scan is redundant -> check_finite=False)
        K = (solve_triangular(R1, Q.T, check_finite=False).T
             if with_k else None)
    else:
        A = core["A"]
        coef, _, _, _ = np.linalg.lstsq(A, D, rcond=rcond)
        Z = A @ coef
        np.subtract(D, Z, out=Z)
        pinvI, _, _, _ = np.linalg.lstsq(A, np.eye(A.shape[0]), rcond=rcond)
        K = pinvI.T
    return Z, K, Z.T @ core["Res"]


def _varproj_factors(E, B, theta, sample_weights, rcond, ridge=0.0):
    """Shared internals: returns (Res, X, Z, K, S).

    Res = A X - Y is the residual in (N(+M), n_ch) layout; X (M, n_ch) the
    LS coefficients; Z = (I-P_A) D the projected partials (N(+M), 2M);
    K = A (A^T A)^{-1} (N(+M), M); S = Z^T Res (2M, n_ch).
    Thin-QR fast path, SVD fallback (rank truncation via rcond) if the QR
    is rank-deficient.

    Split internally into _varproj_core + _varproj_jac so fit_poles can
    skip the Jacobian pieces on cost-only (trial) evaluations. The
    combined result is bit-identical to the monolithic computation.

    With ridge > 0 the design is augmented with lambda * I rows and the
    responses with zeros (the residual rows then also carry the ridge
    penalty lambda * X, so the outer cost is exactly
    ||A X - Y||^2 + lambda^2 ||X||^2 and the Golub-Pereyra derivative
    below remains exact for the augmented design).
    """
    core = _varproj_core(E, B, theta, sample_weights, rcond, ridge=ridge)
    Z, K, S = _varproj_jac(core, E, sample_weights, rcond)
    return core["Res"], core["X"], Z, K, S


class VarProjJacobian(LinearOperator):
    """Exact variable-projection Jacobian as a LinearOperator.

    Each column j is a difference of two rank-1 terms,

        J[:, j] = vec(Z[:, j] x_i^T - K[:, i] S[j, :]),   j = 2i + part,

    so products with J never materialize the (N*n_ch, 2M) dense matrix
    (which reaches hundreds of MB at large p). Row ordering matches
    residual_and_jacobian: index n*n_ch + c.
    """

    def __init__(self, X, Z, K, S):
        N = Z.shape[0]
        n_ch = X.shape[1]
        self.X, self.Z, self.K, self.S = X, Z, K, S
        super().__init__(dtype=float, shape=(N * n_ch, 2 * X.shape[0]))
        self._N, self._n_ch = N, n_ch

    def _matvec(self, v):
        X, Z, K, S = self.X, self.Z, self.K, self.S
        Ve = v[0::2]
        Vw = v[1::2]
        out = (Z[:, 0::2] * Ve[None, :]) @ X
        out += (Z[:, 1::2] * Vw[None, :]) @ X
        out -= K @ (Ve[:, None] * S[0::2] + Vw[:, None] * S[1::2])
        return out.ravel(order="C")

    def _matmat(self, V):
        return np.column_stack([self._matvec(V[:, j])
                                for j in range(V.shape[1])])

    def _rmatvec(self, u):
        X, Z, K, S = self.X, self.Z, self.K, self.S
        Um = u.reshape(self._N, self._n_ch, order="C")
        U = Um @ X.T                      # (N, M)
        Ue = Um @ S[0::2].T              # (N, M)
        Uw = Um @ S[1::2].T              # (N, M)
        out = np.empty(2 * X.shape[0])
        out[0::2] = (Z[:, 0::2] * U).sum(axis=0) - (K * Ue).sum(axis=0)
        out[1::2] = (Z[:, 1::2] * U).sum(axis=0) - (K * Uw).sum(axis=0)
        return out


def _jac_dense_kernel(Z, X, K, S, J):
    """Fallback (no numba): the element formula only. Never called when
    numba is present — the decorated replacement below shadows this."""
    n_rows, twoM = Z.shape
    n_ch = X.shape[1]
    for n in range(n_rows):
        base = n * n_ch
        for c in range(n_ch):
            row = base + c
            for j in range(twoM):
                i = j // 2
                J[row, j] = Z[n, j] * X[i, c] - K[n, i] * S[j, c]


if _njit is not None:
    _jac_dense_kernel = _njit(cache=True)(_jac_dense_kernel)


def _psd_hermitize_kernel(C, out):
    """out[m,i,j] = 0.5 * (C[m,i,j] + conj(C[m,j,i])).
    Elementwise - bitwise-identical to 0.5*(C + conj(swapaxes)),
    one pass straight into the output buffer (no temporaries)."""
    M, p, _ = C.shape
    for m in range(M):
        for i in range(p):
            for j in range(p):
                a = C[m, i, j]
                b = C[m, j, i]
                out[m, i, j] = 0.5 * (a + b.conjugate())


def _psd_reconstruct_kernel(V, lam, out):
    """out[m,i,j] = sum_k V[m,i,k] * lam[m,k] * conj(V[m,j,k]) -
    the PSD eigen-reconstruction (V * diag(lam_pos) * V^H) in ONE
    fused pass: no lam broadcast, no conj(V) copy, no einsum
    internals - a single output allocation. k-ordered reduction
    (gate-checked against the einsum reference; NOT guaranteed
    bitwise - see validation/14_psd_numba_fusion)."""
    M, p, _ = V.shape
    for m in range(M):
        for i in range(p):
            for j in range(p):
                acc = 0.0 + 0.0j
                for k in range(p):
                    acc += (V[m, i, k] * lam[m, k]) * \
                        V[m, j, k].conjugate()
                out[m, i, j] = acc


def _admm_ghat_kernel(Z, U, ia, ib, re, AtY, rho, out):
    """ADMM X-update right-hand side in one pass:
    out[m,c] = AtY[m,c] + rho * ((re[c]) ? (Z-U)[m,ia[c],ib[c]].real
                                        : (Z-U)[m,ia[c],ib[c]].imag)
    Replaces (Z - U), the fancy-index copy, np.where and the
    AtY + rho*G temporaries. Elementwise - bitwise-identical to the
    vectorized sequence."""
    M = Z.shape[0]
    n_ch = ia.shape[0]
    for m in range(M):
        for c in range(n_ch):
            d = Z[m, ia[c], ib[c]] - U[m, ia[c], ib[c]]
            v = d.real if re[c] else d.imag
            out[m, c] = AtY[m, c] + rho * v


def _admm_c_hat_kernel(C, Z, alpha, out):
    """out = alpha * C + (1 - alpha) * Z (over-relaxation), elementwise
    into a reused buffer - bitwise-identical to the vectorized form."""
    M, p, _ = C.shape
    beta = 1.0 - alpha
    for m in range(M):
        for i in range(p):
            for j in range(p):
                out[m, i, j] = alpha * C[m, i, j] + beta * Z[m, i, j]


if _njit is not None:
    _psd_hermitize_kernel = _njit(cache=True)(_psd_hermitize_kernel)
    _psd_reconstruct_kernel = _njit(cache=True)(_psd_reconstruct_kernel)
    _admm_ghat_kernel = _njit(cache=True)(_admm_ghat_kernel)
    _admm_c_hat_kernel = _njit(cache=True)(_admm_c_hat_kernel)


def _assemble_dense_j(Res, X, Z, K, S):
    """Dense Golub-Pereyra Jacobian from factor outputs (shared by
    residual_and_jacobian and fit_poles' cached dense-jac branch)."""
    n_ch = Res.shape[1]
    M = X.shape[0]
    if _njit is not None:
        # numba kernel: writes J directly, no Xr/Kr repeats and no
        # (N, 2M, n_ch) broadcast temporaries or transpose copy.
        # Bitwise identical to the numpy path (strict FP, no FMA).
        J = np.empty((Res.shape[0] * n_ch, 2 * M))
        _jac_dense_kernel(Z, X, K, S, J)
        return J
    # vectorized (guide 4.2): J[n, c, j] = Z[n, j] X[j//2, c]
    #                              - K[n, j//2] S[j, c]
    Xr = np.repeat(X, 2, axis=0)             # (2M, n_ch)
    Kr = np.repeat(K, 2, axis=1)             # (N(+M), 2M)
    J = Z[:, :, None] * Xr[None, :, :] - Kr[:, :, None] * S[None, :, :]
    J = np.ascontiguousarray(J.transpose(0, 2, 1))
    return J.reshape(Res.shape[0] * n_ch, 2 * M)


def residual_and_jacobian(E, B, theta, sample_weights=None, rcond=None,
                          ridge=0.0):
    """Exact variable-projection residual and dense Jacobian (Golub-Pereyra).

    With beta the inner LS solution, the residual R = A beta - Y satisfies

        dR/dtheta_j = (I - P_A) dA_j beta - A (A^T A)^{-1} dA_j^T R

    (the second term vanishes in the cost gradient - Kaufman's
    simplification - but not in the Jacobian itself).

    Returns (r, J): r (N*n_ch,), J (N*n_ch, 2M).
    """
    Res, X, Z, K, S = _varproj_factors(E, B, theta, sample_weights, rcond,
                                       ridge=ridge)
    return Res.ravel(order="C"), _assemble_dense_j(Res, X, Z, K, S)


# ==========================================================================
# SECTION: Gram trust-region solver (tr_solver="gram"; never_form_J_guide)
# ==========================================================================
""""Never form J": a trust-region loop that never materializes the
(N*n_ch x 2M) Jacobian. scipy's trf only uses J through two small
objects, the Gram matrix G = J^T J (2M x 2M) and the gradient
g = J^T f (2M,), and both are built directly from the variable
projection factors (Z, X, S, R1, Q) at a cost INDEPENDENT of the
channel count. The per-iteration SVD of the augmented Jacobian
(scipy 'exact') or the lsmr inner loop is replaced by one symmetric
eigendecomposition of the augmented Gram matrix.

MATHEMATICS. With the Golub-Pereyra Jacobian pieces
    Res = A X - Y                (R, n_ch)   [R = N(+M) with ridge]
    Z = (I - P_A) D               (R, 2M)    projected partials
    K = A (A^T A)^-1              (R, M)
    S = Z^T Res                   (2M, n_ch)
    k(p) = p // 2                 pole index of parameter p
the dense Jacobian (row n*n_ch + c) is
    J[n*n_ch+c, p] = Z[n, p] X[k(p), c] - K[n, k(p)] S[p, c].
Because the columns of Z lie in range(A)^perp while the columns of K
lie in range(A) (Z^T K = 0, measured 1e-15), the cross terms in
J^T J vanish entry by entry, leaving two Hadamard products:
    G[p, q] = (Z^T Z)[p, q] (X X^T)[k(p), k(q)]
            + (K^T K)[k(p), k(q)] (S S^T)[p, q].
On the QR path K = Q R1^-T, so K^T K = R1^-1 R1^-T (M x M, no row sum)
and K^T Res = R1^-1 (Q^T Res); on the rank-truncated SVD fallback
path K is passed explicitly. The gradient is
    g[p] = Z[:, p] . (Res X^T)[:, k(p)] - S[p, :] . (K^T Res)[k(p), :],
(the second term is ~1e-13 relative - Kaufman's simplification - but
kept anyway; it is cheap). Verified against J^T J / J^T f from the
dense Jacobian to 1e-15 with weights and ridge, on both paths
(validation/10_gram_solver/identities.py).

Inside the trust-region step, scipy scales J by the Coleman-Li vector
d: J_h = J diag(d). Everything scipy needs from J_h is quadratic:
    evaluate_quadratic(J_h, g_h, s)  = 0.5 s^T (G_h + diag) s + g_h^T s
    build_quadratic_1d(J_h, ...)     = the same along a line,
with G_h = J_h^T J_h = G * d d^T (elementwise outer(d, d)). The exact
subproblem solve replaces the SVD of [J_h; diag(sqrt(diag_h))] by
    G_aug = G_h + diag(diag_h) = V diag(s^2) V^T    (eigh)
    uf = V_kept^T g_h / s_kept
which is equivalent: J_aug^T f_aug = g_h and J_aug^T J_aug = G_aug,
so U^T f_aug = V^T g_h / s. Directions with s < gram_cut * s[0] are
truncated (eigh only resolves singular values down to ~sqrt(eps) ~
1e-8 of the largest, vs ~1e-12 for the SVD - the price of never
forming J; the trust region clamps those weakly determined directions
anyway). The accept/reject test always uses the TRUE residual cost
0.5 ||f||^2 (ridge rows included); only the step uses G and g.

Caveats (measured, see HANDOFF): the trajectory is NOT bit-identical
to scipy's 'exact' (different rounding, truncated weak directions) -
validate with the relaxed gate (same iteration counts, final cost,
fitted poles). scipy private API is used (scipy.optimize._lsq.common:
solve_lsq_trust_region, CL_scaling_vector, ...) - pinned to scipy
1.18.1; rerun the comparison tests after a scipy upgrade.
"""
try:
    from scipy.optimize._lsq.common import (
        CL_scaling_vector, check_termination, find_active_constraints,
        in_bounds, intersect_trust_region, make_strictly_feasible,
        minimize_quadratic_1d, solve_lsq_trust_region,
        step_size_to_bound, update_tr_radius, print_header_nonlinear,
        print_iteration_nonlinear)
    _TRF_COMMON_OK = True
except ImportError:  # pragma: no cover - tr_solver="gram" unavailable
    _TRF_COMMON_OK = False

# The private-API import succeeded: still warn if the running scipy
# is not the VERIFIED version - a same-named private API on another
# release is untested (results/wall time may differ). The pin lives
# in requirements.txt (scipy==1.18.1); this is the loud fallback.
import scipy as _scipy  # noqa: E402
_SCIPY_VERIFIED = "1.18.1"
if _TRF_COMMON_OK and _scipy.__version__ != _SCIPY_VERIFIED:
    warnings.warn(
        f"tr_solver='gram' verified on scipy {_SCIPY_VERIFIED}, "
        f"running {_scipy.__version__}: the private API "
        "scipy.optimize._lsq.common imported cleanly, but this "
        "combination is UNTESTED - results and wall time may "
        "differ. Pin scipy==1.18.1 (requirements.txt) or use "
        "tr_solver='lsmr'/'exact'.", RuntimeWarning, stacklevel=2)

# Optional diagnostic hook for the gram solver (validation scripts):
# when set to a callable, _trf_gram calls it at every outer iteration
# with a record of the current Gram spectrum (lam, s, keep, n_trunc,
# min(s)/s[0], the live G/g/d/diag_h and x). Leave None in production.
gram_diag_hook = None


def gram_and_grad(Res, X, Z, S, R1=None, Q=None, K=None):
    """G = J^T J (2M x 2M) and g = J^T f without forming J (guide 3.1).

    QR path: pass R1 and Q (K is never needed - K^T K = R1^-1 R1^-T).
    SVD-fallback path: pass K (R1/Q do not exist there)."""
    M = X.shape[0]

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

    k = np.arange(2 * M) // 2
    G = ZZ * XX[np.ix_(k, k)] + KK[np.ix_(k, k)] * SS
    G = 0.5 * (G + G.T)                          # exact symmetry for eigh

    T = Res @ X.T                                # (R, M)
    g = np.empty(2 * M)
    g[0::2] = np.einsum("ni,ni->i", Z[:, 0::2], T)
    g[1::2] = np.einsum("ni,ni->i", Z[:, 1::2], T)
    g[0::2] -= np.einsum("ic,ic->i", S[0::2], KtRes)
    g[1::2] -= np.einsum("ic,ic->i", S[1::2], KtRes)
    return G, g


def _gram_eval_quadratic(G, g, s, diag=None):
    """0.5 * s^T (G + diag) s + g^T s  (1-D s) - Gram form of scipy's
    evaluate_quadratic with J^T J replaced by G."""
    q = s @ (G @ s)
    if diag is not None:
        q += np.dot(s * diag, s)
    return 0.5 * q + np.dot(g, s)


def _gram_build_quadratic_1d(G, g, s, diag=None, s0=None):
    """Gram form of scipy's build_quadratic_1d:
    f(t) = 0.5 (s0 + s t)^T (G + diag) (s0 + s t) + g^T (s0 + s t)."""
    v = G @ s
    a = np.dot(s, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5
    b = np.dot(g, s)
    if s0 is not None:
        u = G @ s0
        b += np.dot(s0, v)
        c = 0.5 * np.dot(s0, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c
    return a, b


def _gram_select_step(x, G_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    """scipy's trf select_step with J_h replaced by G_h = J_h^T J_h
    (uses _gram_eval_quadratic / _gram_build_quadratic_1d)."""
    if in_bounds(x + p, lb, ub):
        p_value = _gram_eval_quadratic(G_h, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict trust-region step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Reflected direction will cross first either feasible region or
    # trust region boundary.
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    if r_stride_l <= r_stride_u:
        a, b, c = _gram_build_quadratic_1d(G_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p *= theta
    p_h *= theta
    p_value = _gram_eval_quadratic(G_h, g_h, p_h, diag=diag_h)

    ag_h = -g_h
    ag = d * ag_h

    to_tr = Delta / norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr

    a, b = _gram_build_quadratic_1d(G_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


def _trf_gram(fun, gram, x0, lb, ub, ftol=1e-8, xtol=1e-8, gtol=1e-8,
              max_nfev=None, x_scale="jac", verbose=0, cut=1e-9,
              context=None):
    """Vendored scipy trf_bounds (scipy 1.18.1) where the Jacobian
    callback `gram(x) -> (G, g)` replaces the matrix: G = J^T J,
    g = J^T f. Structure, termination logic, trust-radius updates and
    nfev accounting follow scipy exactly so iteration counts stay
    comparable with least_squares; only the linear algebra differs
    (eigh(G_aug) instead of svd(J_augmented), Gram-form quadratics).

    fun must return the TRUE full residual f (ridge rows included) -
    the accept/reject test always uses 0.5 ||f||^2. x_scale: "jac"
    (column norms from diag(G), scipy's zero-guard + never-decrease
    rule) or a fixed positive array. cut: relative eigenvalue
    truncation floor (directions with s < cut * s[0] are dropped).

    Returns an OptimizeResult like least_squares (x, cost, fun, grad,
    optimality, nfev, njev, status, message, active_mask) plus
    n_trunc: total truncated directions across accepted iterations.
    context: opaque object passed through to the gram_diag_hook
    record (fit_poles passes its factor cache).
    """
    if not _TRF_COMMON_OK:
        raise RuntimeError(
            "tr_solver='gram' needs scipy.optimize._lsq.common (private "
            "API, verified on scipy 1.18.1); this scipy moved/renamed it. "
            "Use tr_solver='lsmr' or 'exact'.")

    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    if not in_bounds(x0, lb, ub):
        raise ValueError("`x0` is infeasible.")
    x = make_strictly_feasible(x0, lb, ub)

    f = fun(x)
    if not np.all(np.isfinite(f)):
        raise ValueError("Residuals are not finite in the initial point.")
    f_true = f.copy()
    nfev = 1

    G, g = gram(x)
    njev = 1
    n = x.size
    m_rows = f.size
    cost = 0.5 * np.dot(f, f)

    jac_scale = isinstance(x_scale, str) and x_scale == "jac"
    if jac_scale:
        scale_inv = np.sqrt(np.maximum(np.diag(G), 0.0))
        scale_inv[scale_inv == 0] = 1.0
    else:
        scale_inv = np.asarray(x_scale, dtype=float)
        if np.any(scale_inv <= 0):
            raise ValueError("`x_scale` must be 'jac' or a positive "
                             "array.")
    scale = 1.0 / scale_inv

    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x * scale_inv / v**0.5)
    if Delta == 0:
        Delta = 1.0

    g_norm = norm(g * v, ord=np.inf)

    if max_nfev is None:
        max_nfev = n * 100

    alpha = 0.0          # Levenberg-Marquardt parameter
    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None
    n_trunc = 0

    if verbose == 2:
        print_header_nonlinear()

    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)

        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost,
                                      actual_reduction, step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        # "hat"-space quantities (x_scale then Coleman-Li), exactly as
        # scipy's trf_bounds:
        v[dv != 0] *= scale_inv[dv != 0]
        d = v**0.5 * scale
        diag_h = g * dv * scale
        g_h = d * g
        G_h = G * np.outer(d, d)              # J_h^T J_h
        G_aug = G_h + np.diag(diag_h)        # + Coleman-Li term
        G_aug = 0.5 * (G_aug + G_aug.T)
        lam, V = np.linalg.eigh(G_aug)       # dsyevd: fastest driver
        lam = lam[::-1]
        V = V[:, ::-1]
        s = np.sqrt(np.maximum(lam, 0.0))
        keep = s > cut * s[0]
        n_trunc += n - int(keep.sum())
        if not np.any(keep):
            # degenerate: G_aug == 0 implies g == 0 (terminated above);
            # guard anyway with a zero step.
            keep[0] = True
            s[0] = max(s[0], np.finfo(float).tiny)
        sk = s[keep]
        Vk = V[:, keep]
        uf = (Vk.T @ g_h) / sk

        if gram_diag_hook is not None:
            # diagnostic hook (validation/10_gram_solver): called at
            # EVERY outer iteration with the current spectrum, so
            # scripts can log min(s)/s[0] and the truncated-direction
            # count along the whole trajectory (never_form_J follow-up
            # question: WHERE does the cut matter?). `context` is the
            # fit_poles factor cache (live dict with "core" and "jac":
            # Res/X/Q/R1 and Z/K/S of the CURRENT x) so hooks can
            # rebuild J and cross-check the weak eigenpairs against
            # svd(J_aug). All references are live - consume
            # immediately, do not store.
            gram_diag_hook(dict(
                iteration=iteration, x=x.copy(), cost=cost, n=n,
                G=G, g=g, d=d, diag_h=diag_h, Delta=Delta,
                lam=lam, V=V, s=s, keep=keep, cut=cut,
                n_trunc=n - int(keep.sum()), context=context))

        # theta controls step-back ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            p_h, alpha, _n_iter = solve_lsq_trust_region(
                int(keep.sum()), m_rows, uf, sk, Vk, Delta,
                initial_alpha=alpha)

            p = d * p_h  # trust-region solution in the original space.
            step, step_h, predicted_reduction = _gram_select_step(
                x, G_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio,
                ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new
            f = f_new
            f_true = f.copy()
            cost = cost_new

            G, g = gram(x)
            njev += 1

            if jac_scale:
                new_inv = np.sqrt(np.maximum(np.diag(G), 0.0))
                new_inv[new_inv == 0] = 1.0
                scale_inv = np.maximum(new_inv, scale_inv)
                scale = 1.0 / scale_inv
        else:
            step_norm = 0.0
            actual_reduction = 0.0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    if termination_status == 0:
        message = ("The maximum number of function evaluations is "
                   "exceeded.")
    elif termination_status == 1:
        message = "`gtol` termination condition is satisfied."
    elif termination_status == 2:
        message = "`ftol` termination condition is satisfied."
    elif termination_status == 3:
        message = "`xtol` termination condition is satisfied."
    else:
        message = ("Both `ftol` and `xtol` termination conditions are "
                   "satisfied.")

    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=None, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status, message=message, n_trunc=n_trunc,
        nit=iteration)


def default_bounds(E, e_margin=0.05, w_min=None, w_max=2.5):
    """Bounds factory for theta in scaled units: centers near the window,
    widths strictly positive. Returns callable(n_pairs) -> (lo, hi).

    w_min defaults to max(1e-3, 0.15 * median sample spacing): features
    far below the sample spacing are invisible to the samples (their LS
    columns are nearly parallel), so unpenalized sub-resolution poles only
    produce between-sample/off-window artifacts. 0.15 * spacing forbids
    the observed artifact pairs (~0.075 * spacing) while leaving narrow
    but still-sampled resonances free.
    """
    E = np.asarray(E, dtype=float)
    Emin, Emax = float(E.min()), float(E.max())
    span = max(Emax - Emin, 1e-12)
    if w_min is None:
        dE = float(np.median(np.diff(np.sort(E)))) if E.size > 1 else span
        w_min = max(1e-3, 0.15 * dE)

    def bounds_for(n_pairs):
        lo = np.empty(2 * n_pairs)
        hi = np.empty(2 * n_pairs)
        lo[0::2] = Emin - e_margin * span
        hi[0::2] = Emax + e_margin * span
        lo[1::2] = w_min
        hi[1::2] = w_max
        return lo, hi

    return bounds_for


def compress_channels(B, rank_tol=1e-9, min_gain=0.75):
    """Rotate B onto its strongest singular directions (performance
    guide 4.1).

    All channels share the same poles, so the response matrix
    B (n_ch, N) is numerically LOW RANK for smooth level-width data
    (measured on real Ag3Au3 Gammas: numerical rank 9-19 of 729
    channels at 1e-9-1e-4 relative). The LS cost is invariant under
    orthogonal rotation of the channel axis, so the optimizer can run
    on the r strongest directions: rows drop from N*n_ch to N*r in
    the Jacobian and its products. The FINAL coefficient solve always
    runs against the full responses, so the returned X/R are exact.

    rank_tol : singular-value cutoff relative to s[0]. None disables.
    min_gain : skip compression when fewer than ~25% of the rows are
        saved (not worth the SVD).
    Returns Bc (r, N), or B itself when compression does not pay.
    """
    B = np.asarray(B)
    if not rank_tol or B.shape[0] < 2:
        return B
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    if s[0] <= 0.0:
        return B
    r = int(np.sum(s > rank_tol * s[0]))
    r = max(r, 1)
    if r >= min_gain * B.shape[0]:
        return B
    # Bc = U_r^T B  (rows of Vt scaled by the singular values)
    return s[:r, None] * Vt[:r]


def fit_poles(E, B, centers0, widths0, sample_weights=None,
              bounds_fun=None, method="trf", max_nfev=None, rcond=None,
              verbose=0, ridge=0.0, rank_tol=1e-9, B_opt=None,
              ftol=1e-8, xtol=1e-8, gtol=1e-8,
              operator_threshold=3_000_000, tr_solver="lsmr",
              gram_cut=1e-9):
    """Outer nonlinear fit over (e_i, w_i) with coefficients re-solved
    at every evaluation (variable projection).

    E : (N,) scaled energies; B : (n_ch, N) responses
    centers0, widths0 : initial poles (scaled units)
    bounds_fun : callable(n_pairs) -> (lo, hi) for theta
    B_opt : pre-compressed responses (guide 2.1) - the compression SVD
        need not be repeated for every fit_poles call within one
        _fit_impl run (B is constant there). Must be compress_channels
        of THIS B.
    ridge : Tikhonov penalty on the coefficients (see solve_coefficients)
    tr_solver : 'exact' (dense SVD of J per step), 'lsmr' (inexact,
        faster at M >~ 40, edge-band bias) or 'gram' - the "never form
        J" solver (never_form_J_guide): the loop only sees
        G = J^T J / g = J^T f built from the varproj factors (cost
        independent of the channel count) and one eigh per step. See
        the Gram solver section above for the math and caveats.
    gram_cut : relative eigenvalue truncation floor of the 'gram'
        solver (default 1e-9; directions with s < gram_cut * s[0]
        dropped). 1e-9 keeps genuinely weak directions that 1e-7
        wrongly truncates (measured on Ag3Au3: 1e-7 degrades to
        min eig -1.6e5; 1e-9/1e-11 identical and healthy) - see the
        gram_cut entry in _fit_impl for the full measurement.

    Returns dict with keys e, w, X, R (weighted residuals), cost, nfev.
    """
    centers0 = np.asarray(centers0, dtype=float)
    widths0 = np.asarray(widths0, dtype=float)
    theta0 = np.empty(2 * centers0.size)
    theta0[0::2] = centers0
    theta0[1::2] = widths0

    # channel compression (guide 4.1): the OPTIMIZER runs on the
    # strongest channel directions; the final solve below is against
    # the FULL responses, so X/R are exact. B_opt skips the SVD when
    # the caller already compressed this B (guide 2.1).
    B_full = np.asarray(B)
    B = B_opt if B_opt is not None else compress_channels(B_full,
                                                          rank_tol)

    # one-entry factor cache (guides 4.3 + A1): scipy calls fun at trial
    # points (cost only) and fun+jac at accepted points. The CORE (QR,
    # X, Res) is computed once per theta; the Jacobian pieces (D, Z, K,
    # S) are computed lazily and memoized with the core. Exact, not
    # approximate - same ops, same order as computing everything.
    _cache = {"key": None, "core": None, "jac": None}

    def core(theta):
        key = theta.tobytes()
        if _cache["key"] != key:
            _cache["core"] = _varproj_core(E, B, theta, sample_weights,
                                           rcond, ridge=ridge)
            _cache["key"] = key
            _cache["jac"] = None
        return _cache["core"]

    def factors(theta):
        c = core(theta)
        if _cache["jac"] is None:
            _cache["jac"] = _varproj_jac(c, E, sample_weights, rcond)
        Z, K, S = _cache["jac"]
        return c["Res"], c["X"], Z, K, S

    def fun(theta):
        return core(theta)["Res"].ravel(order="C")

    if tr_solver == "gram":
        # "never form J" (never_form_J_guide): the optimizer only ever
        # sees (G, g) built from the factors - no dense J, no operator,
        # no per-step SVD/lsmr. K is skipped on the QR path (with_k=
        # False); the SVD fallback returns it and gram_and_grad uses
        # K^T K there.
        def gram(theta):
            c = core(theta)
            if _cache["jac"] is None:
                _cache["jac"] = _varproj_jac(c, E, sample_weights, rcond,
                                             with_k=False)
            Z, K, S = _cache["jac"]
            if c["svd_path"]:
                return gram_and_grad(c["Res"], c["X"], Z, S, K=K)
            return gram_and_grad(c["Res"], c["X"], Z, S,
                                 R1=c["R1"], Q=c["Q"])

        if bounds_fun is not None:
            lo, hi = bounds_fun(centers0.size)
        else:
            lo = np.full(theta0.size, -np.inf)
            hi = np.full(theta0.size, np.inf)
        res = _trf_gram(fun, gram, theta0, lo, hi, ftol=ftol, xtol=xtol,
                        gtol=gtol, max_nfev=max_nfev, x_scale="jac",
                        verbose=verbose, cut=gram_cut, context=_cache)

        e = res.x[0::2]
        w = res.x[1::2]
        Phi = lorentzian_basis(E, e, w)
        X, R = solve_coefficients(Phi, B_full, sample_weights, rcond=rcond,
                                  ridge=ridge)
        return dict(e=e, w=w, X=X, R=R, cost=float(res.cost),
                    nfev=int(res.nfev), status=int(res.status),
                    message=res.message)

    # the dense Jacobian is (N*n_ch, 2M); above ~3e6 entries switch to a
    # rank-1-structured LinearOperator (same matrix, no materialization)
    N = np.asarray(E).size
    n_ch = np.asarray(B).shape[0]
    dense_size = N * n_ch * 2 * centers0.size

    if dense_size > operator_threshold:
        def jac(theta):
            Res, X, Z, K, S = factors(theta)
            return VarProjJacobian(X, Z, K, S)

        # column norms of J at theta0 (exact, cheap from the rank-1 form):
        # ||J_j||^2 = |z_j|^2 |x_i|^2 + |k_i|^2 |s_j|^2 - 2 (z_j.k_i)(x_i.s_j)
        Res, X, Z, K, S = factors(theta0)
        zz = np.sum(Z * Z, axis=0)                       # (2M,)
        ss = np.sum(S * S, axis=1)                       # (2M,)
        x2 = np.sum(X * X, axis=1)                       # (M,)
        k2 = np.sum(K * K, axis=0)                       # (M,)
        ZK = Z.T @ K                                     # (2M, M)
        XS = X @ S.T                                     # (M, 2M)
        col2 = zz * np.repeat(x2, 2) + np.repeat(k2, 2) * ss
        col2[0::2] -= 2.0 * np.diag(ZK[0::2]) * np.diag(XS[:, 0::2])
        col2[1::2] -= 2.0 * np.diag(ZK[1::2]) * np.diag(XS[:, 1::2])
        x_scale = np.sqrt(np.maximum(col2, 1e-300))
        x_scale = np.maximum(x_scale, 1e-12 * np.max(x_scale))
    else:
        def jac(theta):
            # cached core + lazily computed jac pieces: no recomputation
            # of the QR block (the old path called
            # residual_and_jacobian, which redid _varproj_factors)
            Res, X, Z, K, S = factors(theta)
            return _assemble_dense_j(Res, X, Z, K, S)
        x_scale = "jac"

    kwargs = dict(x_scale=x_scale, method=method, jac=jac,
                  ftol=ftol, xtol=xtol, gtol=gtol)
    if tr_solver is not None:
        # None = scipy default ('exact' for dense jac). 'lsmr' avoids
        # the per-iteration dense SVD (speedupguide_new.md step 5) at
        # the cost of a different optimizer trajectory.
        kwargs["tr_solver"] = tr_solver
    if bounds_fun is not None:
        lo, hi = bounds_fun(centers0.size)
        kwargs["bounds"] = (lo, hi)
    if max_nfev is not None:
        kwargs["max_nfev"] = max_nfev
    if verbose:
        kwargs["verbose"] = verbose

    res = least_squares(fun, theta0, **kwargs)

    e = res.x[0::2]
    w = res.x[1::2]
    Phi = lorentzian_basis(E, e, w)
    # final solve against the FULL responses: X is (M, n_ch), R is
    # (n_ch, N) - exact, compression only ever touched the optimizer
    X, R = solve_coefficients(Phi, B_full, sample_weights, rcond=rcond,
                              ridge=ridge)
    return dict(e=e, w=w, X=X, R=R, cost=float(res.cost), nfev=int(res.nfev),
                status=int(res.status), message=res.message)


def basis_apply(Phi, C):
    """Model on a real grid from basis and residues: (N, M) x (M, p, p)
    -> (N, p, p). Same as einsum('ni,ijk->njk') but as a single BLAS
    matmul (guide 4.8) - this runs per candidate in the adaptive loop."""
    Phi = np.asarray(Phi)
    C = np.asarray(C)
    M, p, _ = C.shape
    return (Phi @ C.reshape(M, p * p)).reshape(Phi.shape[0], p, p)


# ==========================================================================
# SECTION: poles_init
# ==========================================================================
"""Pole initialization for the shared-pole Lorentzian fit.

Two strategies:
- linspace: M centers spread over the energy window, widths a few percent
  of the window. Robust, no dependencies.
- skrf (option A of the implementation plan): run scikit-rf's VectorFitting
  pole relocation on the stacked matrix entries and adopt its fitted
  conjugate pole pairs as starting (e_i, w_i); residues are discarded.

scikit-rf convention mapping
----------------------------
skrf fits G(s) = sum_k [r_k/(s - p_k) + conj(r_k)/(s - conj(p_k))] with
s = 1j * 2*pi * f, storing one pole per conjugate pair (Im >= 0) and
flipping unstable poles to Re <= 0.

We feed the energies as positive frequencies f = E + sigma. A Lorentzian
pair e +/- i*w of our basis then corresponds to the s-domain poles
{-2*pi*w + 1j*2*pi*(e+sigma), +2*pi*w + 1j*2*pi*(e+sigma)}: the first is
skrf-stable and is exactly the stored pole; the second is flipped by skrf.
The stored conjugate partner conj(p) maps back to a mirrored, far-off
center (-e - 2*sigma) that only contributes smooth background over the
window. Extraction is therefore one line per stored pole:

    e = Im(p)/(2*pi) - sigma,     w = -Re(p)/(2*pi)   (Re(p) <= 0)
"""

import sys
import warnings

import numpy as np


def linspace_init(E, n_pairs, width_frac=0.02):
    """Centers spread over the window, widths = width_frac * span."""
    E = np.asarray(E, dtype=float)
    span = max(float(E.max() - E.min()), 1e-12)
    centers = np.linspace(E.min(), E.max(), n_pairs)
    widths = np.full(n_pairs, width_frac * span)
    return centers, widths


def _dedupe(centers, widths, e_tol, w_tol_rel):
    """Merge near-identical pairs (average), preserving order."""
    if centers.size == 0:
        return centers, widths
    keep_e, keep_w, counts = [], [], []
    for e, w in zip(centers, widths):
        merged = False
        for k in range(len(keep_e)):
            if (abs(e - keep_e[k]) < e_tol
                    and abs(w - keep_w[k]) < w_tol_rel * (w + keep_w[k] + 1e-30)):
                keep_e[k] = (keep_e[k] * counts[k] + e) / (counts[k] + 1)
                keep_w[k] = (keep_w[k] * counts[k] + w) / (counts[k] + 1)
                counts[k] += 1
                merged = True
                break
        if not merged:
            keep_e.append(float(e))
            keep_w.append(float(w))
            counts.append(1)
    return np.array(keep_e), np.array(keep_w)


def skrf_init(E, F, n_pairs, e_margin=0.25, width_floor_frac=1e-3,
              max_ports=32, verbose=False):
    """VectorFitting pole relocation on the stacked matrix entries.

    Math: the Sanathanan-Koerner iteration for rational fitting. The
    model  sum_i C_i/(s - p_i)  is multiplied through by the common
    denominator D(s) = prod_i (s - p_i), giving the LINEAR-in-poles
    form  F(s) D(s) = sum_i C_i prod_{k != i} (s - p_k); one weighted
    complex LS per iteration relocates the poles p_i (we keep only
    the pole locations — our own model refits the residues). skrf's
    implementation runs that iteration on the stacked S-parameters.

    E : (N,) scaled energies; F : (N, p, p) scaled Hermitian samples.
    Returns (centers, widths) in scaled units, length <= n_pairs.

    Poles whose relocated center falls outside the window (plus margin)
    are dropped; widths are floored at width_floor_frac * span.

    For large p the relocation LS is O(p^2 N) rows; since all matrix
    entries share the same rational pole structure, the relocation is run
    on the principal submatrix of the `max_ports` strongest diagonal
    entries (residues differ per entry, poles do not).
    """
    try:
        import skrf
        from skrf.vectorFitting import VectorFitting
    except ImportError as exc:
        raise ImportError("scikit-rf is required for init='skrf'") from exc

    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    N = E.size
    span = max(float(E.max() - E.min()), 1e-12)
    # place the window strictly on the positive frequency axis
    sigma = 0.5 * span - float(E.min())
    f = E + sigma
    assert f.min() > 0

    p = F.shape[1]
    F_fit_input = F
    if p > max_ports:
        dnorm = np.sqrt(np.sum(np.abs(F[:, range(p), range(p)]) ** 2, axis=0))
        idx = np.argsort(-dnorm)[:max_ports]
        F_fit_input = F[:, idx[:, None], idx[None, :]]
        if verbose:
            print(f"[skrf_init] p={p} > max_ports={max_ports}; "
                  f"relocating on the {max_ports} strongest diagonal ports")

    freq = skrf.Frequency.from_f(f, unit="Hz")
    nw = skrf.Network(frequency=freq, s=np.asarray(F_fit_input))
    vf = VectorFitting(nw)
    # relocation iterations are a big complex lstsq each; 10 balances
    # initialization quality against runtime at large p (default 100
    # dominates the runtime at large p; over-iterating was observed to
    # COLLAPSE the relocated poles toward the window center on
    # well-resolved synthetic data)
    vf.max_iterations = 10
    with warnings.catch_warnings():
        # relocation warnings are fine: we only want the pole locations
        warnings.simplefilter("ignore")
        vf.vector_fit(n_poles_real=0, n_poles_cmplx=int(n_pairs),
                      init_pole_spacing="lin", fit_constant=False,
                      fit_proportional=False, enforce_dc=False)

    poles = np.atleast_1d(np.asarray(vf.poles, dtype=complex))
    if verbose:
        print(f"[skrf_init] relocated poles (s-domain): {poles}")

    centers = poles.imag / (2 * np.pi) - sigma
    widths = -poles.real / (2 * np.pi)

    in_window = (centers > E.min() - e_margin * span) & \
                (centers < E.max() + e_margin * span)
    centers, widths = centers[in_window], widths[in_window]
    widths = np.maximum(widths, width_floor_frac * span)

    centers, widths = _dedupe(centers, widths, e_tol=0.02 * span,
                             w_tol_rel=0.25)
    return centers, widths


def psd_dict_init(E, F, n_pairs, w_frac=1.5, verbose=False):
    """Positivity-native pole initialization (review item 4.7).

    Tr Gamma(E) >= 0 for physical data, so the trace is a non-negative
    scalar function. A NON-NEGATIVE Lorentzian dictionary fit of the
    trace (scipy nnls) yields a SPARSE pole set natively compatible
    with PSD residues - no inter-pole cancellation needed, which is
    exactly what a PSD model cannot use. Dictionary: centers every
    half local spacing, widths w_frac x local spacing.
    Returns (centers, widths) of the strongest atoms, length <= n_pairs.
    """
    from scipy.optimize import nnls

    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    tr = np.real(np.trace(F, axis1=1, axis2=2))
    N = E.size
    span = max(float(E.max() - E.min()), 1e-12)

    # local spacing at arbitrary positions (linear interp of the
    # sorted-sample spacing)
    Es = np.sort(E)
    ds = np.diff(Es)
    ds = np.maximum(ds, 1e-12)
    spacing = 0.5 * (np.concatenate([[ds[0]], ds]) +
                     np.concatenate([ds, [ds[-1]]]))

    def local_spacing(x):
        return float(np.interp(x, Es, spacing))

    # dictionary: ~2 atoms per sample (every half spacing), capped
    K = int(min(max(4 * N, 8), 4 * n_pairs + 2 * N, 400))
    centers_d = np.linspace(E.min(), E.max(), K)
    widths_d = np.array([w_frac * local_spacing(c) for c in centers_d])
    A = (widths_d[None, :] ** 2 /
         ((E[:, None] - centers_d[None, :]) ** 2 +
          widths_d[None, :] ** 2))
    # weight rows by the local spacing so dense clusters don't dominate
    row_w = np.interp(E, Es, spacing)
    A = A * row_w[:, None]
    b = tr * row_w
    alpha, _ = nnls(A, b)
    keep = np.where(alpha > 1e-12 * max(alpha.max(), 1e-300))[0]
    if keep.size == 0:
        return np.array([]), np.array([])
    # strongest atoms first
    keep = keep[np.argsort(-alpha[keep])][:max(2, n_pairs)]
    centers = centers_d[keep]
    widths = widths_d[keep]
    if verbose:
        print(f"[psd_dict_init] nnls kept {keep.size} of {K} atoms "
              f"(top {n_pairs} requested)")
    return centers, widths


def init_poles(E, F, n_pairs, method="skrf", verbose=False, **kwargs):
    """Dispatch to the selected initialization strategy.

    Falls back to linspace when skrf is unavailable or relocates to an
    empty/degenerate set, so callers always get a usable starting grid.
    The fallback reason is reported on stderr (never silent).
    """
    if method == "skrf":
        try:
            centers, widths = skrf_init(E, F, n_pairs, **kwargs)
            if centers.size >= 2:
                return centers, widths
            print("[poles_init] skrf relocation degenerate; "
                  "falling back to linspace", file=sys.stderr)
        except ImportError:
            print("[poles_init] scikit-rf unavailable; "
                  "falling back to linspace", file=sys.stderr)
        except Exception as exc:
            print(f"[poles_init] skrf relocation failed ({exc!r}); "
                  "falling back to linspace", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
    return linspace_init(E, n_pairs)


# ==========================================================================
# SECTION: prune
# ==========================================================================
"""Pruning, duplicate merging, and residual-driven pole insertion.

The Lorentzian basis decays term by term, but near-cancelling distant or
duplicate pairs violate uniform decay in the complex plane even though
each term decays. Pruning is the enforcement mechanism - do not skip it.
"""

import numpy as np
from scipy.signal import find_peaks


def distance_to_nearest_sample(E, x):
    """Distance from each point in x to the nearest sample in E.

    The natural local-resolution scale of a sampling grid: a pole whose
    width is far below this distance is invisible to every sample near
    it (its basis column is ~0 at all of them) and only produces
    between-sample spikes. On a uniform grid this equals half the
    spacing, recovering the classic 'half the sample spacing' floor.
    """
    E = np.sort(np.asarray(E, dtype=float))
    x = np.asarray(x, dtype=float)
    j = np.clip(np.searchsorted(E, x), 0, E.size)
    left = E[np.clip(j - 1, 0, E.size - 1)]
    right = E[np.clip(j, 0, E.size - 1)]
    return np.minimum(np.abs(x - left), np.abs(x - right))


def subresolution_mask(E, centers, widths, frac=1.0):
    """Boolean mask of poles narrower than frac * their local spacing."""
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if centers.size == 0:
        return np.zeros(0, dtype=bool)
    return widths < frac * distance_to_nearest_sample(E, centers)


def residue_norms(C):
    """||C_i||_F per pole pair. C: (M, p, p)."""
    return np.sqrt(np.sum(np.abs(C) ** 2, axis=(1, 2)))


def prune(centers, widths, C, rel_tol=1e-3, dup_e_tol=0.02, dup_w_tol_rel=0.5):
    """Remove negligible and duplicate pole pairs.

    rel_tol : drop pair i if ||C_i||_F < rel_tol * max_j ||C_j||_F
    dup_e_tol, dup_w_tol_rel : merge pairs closer than this (scaled units;
        duplicate detection catches near-cancellation).

    Returns (centers, widths, C) of the kept pairs.
    """
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    C = np.asarray(C)
    if centers.size == 0:
        return centers, widths, C

    norms = residue_norms(C)
    keep = norms > rel_tol * max(norms.max(), 1e-300)
    centers, widths, C, norms = centers[keep], widths[keep], C[keep], norms[keep]

    order = np.argsort(-norms)  # strongest first, merge into it
    keep_idx = []
    for i in order:
        dup = False
        for j in keep_idx:
            if (abs(centers[i] - centers[j]) < dup_e_tol and
                    abs(widths[i] - widths[j]) <
                    dup_w_tol_rel * (widths[i] + widths[j])):
                dup = True
                break
        if not dup:
            keep_idx.append(int(i))
    keep_idx = np.array(sorted(keep_idx), dtype=int)
    return centers[keep_idx], widths[keep_idx], C[keep_idx]


def per_sample_residual_norms(R):
    """R: (n_ch, N) residuals -> (N,) per-sample norm over channels.

    Off-diagonal entries appear as two channels (re + im) but only once
    (the conjugate is not repeated), so this is proportional to the
    Frobenius norm of the full-matrix residual up to a fixed factor.
    """
    return np.sqrt(np.sum(np.asarray(R) ** 2, axis=0))


def suggest_pole_insertions(E, R, centers, widths, max_insert=1,
                       min_spacing_frac=1.0, prominence_frac=0.05):
    """Residual-driven insertion points: worst residual peaks not already
    covered by an existing pole. Returns arrays of (center, width)."""
    E = np.asarray(E, dtype=float)
    r = per_sample_residual_norms(R)
    if r.max() <= 0:
        return np.array([]), np.array([])
    span = float(E.max() - E.min())

    peaks, props = find_peaks(r, prominence=prominence_frac * r.max())
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(r))])

    # sub-sample peak position by parabolic fit through the peak triplet
    def refine(i):
        if 0 < i < r.size - 1:
            y0, y1, y2 = r[i - 1], r[i], r[i + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-300:
                shift = 0.5 * (y0 - y2) / den
                if abs(shift) <= 1:
                    # local half-spacing (non-uniform grids)
                    return E[i] + shift * 0.5 * (E[i + 1] - E[i - 1])
        return E[i]

    cand = sorted(peaks, key=lambda i: -r[i])
    out_c, out_w = [], []
    for i in cand:
        c = refine(i)
        # local spacing around the peak
        j = min(max(int(i), 1), E.size - 2)
        local_d = 0.5 * ((E[j + 1] - E[j]) + (E[j] - E[j - 1]))
        if centers.size and np.min(np.abs(centers - c)) < \
                min_spacing_frac * max(local_d, 1e-12):
            continue
        out_c.append(c)
        out_w.append(max(local_d, 1e-3 * span))
        if len(out_c) >= max_insert:
            break
    return np.array(out_c), np.array(out_w)


def merge_duplicates(centers, widths, C, e_factor=0.05, w_factor=0.05,
                     cancel_frac=0.15):
    """Merge near-coincident pole pairs with near-cancelling residues.

    Two poles that nearly coincide are a null direction of the sample LS:
    opposite residues cancel on the sampled energies but leak enormous
    spurious structure between samples and off-window. Merging such a pair
    (residues added, position = residue-norm-weighted mean) changes the
    model on the samples by the cancellation error only - which is exactly
    the artifact being removed.

    A pair is merged only if BOTH hold:
    - coincidence: |de| < e_factor * min(w_i, w_j) and
      |dw| < w_factor * (w_i + w_j) / 2   (default 5% overlap)
    - cancellation: ||C_i + C_j|| < cancel_frac * (||C_i|| + ||C_j||)
      (default 0.5: aligned coincident poles are a harmless degeneracy and
      left alone; legitimate close-but-distinct resonances never cancel)

    Returns (centers, widths, C, n_merged) of the merged set.
    """
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    C = np.asarray(C)
    if centers.size == 0:
        return centers, widths, C, 0
    norm = residue_norms(C)
    order = np.argsort(-norm)

    out_e, out_w, out_C, out_n = [], [], [], []
    for i in order:
        target = None
        for k in range(len(out_e)):
            coincident = (abs(centers[i] - out_e[k]) <
                          e_factor * min(widths[i], out_w[k]) and
                          abs(widths[i] - out_w[k]) <
                          w_factor * (widths[i] + out_w[k]) / 2.0)
            if not coincident:
                continue
            ssum = float(out_n[k] + norm[i])
            norm_sum = np.linalg.norm(out_C[k] + C[i])
            if norm_sum < cancel_frac * ssum:
                target = k
                break
        if target is None:
            out_e.append(float(centers[i]))
            out_w.append(float(widths[i]))
            out_C.append(C[i].copy())
            out_n.append(float(norm[i]))
        else:
            k = target
            tot = out_n[k] + norm[i]
            out_e[k] = (out_e[k] * out_n[k] + centers[i] * norm[i]) / tot
            out_w[k] = (out_w[k] * out_n[k] + widths[i] * norm[i]) / tot
            out_C[k] = out_C[k] + C[i]
            out_n[k] = tot
    n_before = centers.size
    return (np.array(out_e), np.array(out_w), np.array(out_C),
            n_before - len(out_e))


# ==========================================================================
# SECTION: constraints
# ==========================================================================
"""Structure constraints on the residue matrices: Hermitian by
construction, PSD by projection + non-negative shared-scale refit.

Order of operations (implementation plan, caveat 4): get poles
unconstrained first, then do the PSD coefficient solve, then optionally one
projected refinement pass. Unconstrained fits rely on inter-pole
cancellation, so per-pole PSD projection during the pole fit is not done.
"""

import numpy as np
from scipy.optimize import nnls


def project_psd(C):
    """Clip negative eigenvalues of each Hermitian C_i to zero.

    C : (M, p, p) Hermitian. Returns (C_pos, n_clipped, worst_eig).
    Batched: one stacked LAPACK call instead of M Python-level eighs
    (this runs inside every ADMM iteration; the loop version dominated
    the PSD solve time at large M).

    numba path (validation/14_psd_numba_fusion): the Hermitization
    writes straight into its buffer (no 0.5*(C + conj(swapaxes))
    temporaries) and the eigen-reconstruction is one fused kernel -
    no lam broadcast, no conj(V) copy, no einsum internals: ONE
    output allocation instead of ~4 per call, ~4000 calls per
    production fit. The hermitize kernel is bitwise-identical to the
    numpy formula; the reconstruction is k-order-reduced (not
    guaranteed bitwise against the einsum - gate-measured).
    """
    C = np.asarray(C)
    M, p, _ = C.shape
    if _njit is not None:
        Her = np.empty_like(C)
        _psd_hermitize_kernel(C, Her)
    else:
        Her = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))
    lam, V = np.linalg.eigh(Her)              # batched (M, p), (M, p, p)
    worst = float(lam.min())
    lam_pos = np.maximum(lam, 0.0)
    n_clipped = int((lam.min(axis=1) < 0).sum())
    if _njit is not None:
        C_pos = np.empty_like(C)
        _psd_reconstruct_kernel(V, lam_pos, C_pos)
    else:
        C_pos = np.einsum("mik,mk,mjk->mij", V, lam_pos, np.conj(V))
    return C_pos, n_clipped, worst


def _channel_values(C_pos, chans):
    """(n_ch, M) real values of each Hermitian C_pos_i along the channels."""
    vals = np.empty((len(chans), C_pos.shape[0]))
    for c, (a, b, part) in enumerate(chans):
        v = C_pos[:, a, b]
        vals[c] = v.real if part == "re" else v.imag
    return vals


def refit_scales(Phi, B, C_pos, chans):
    """Refit non-negative shared scales alpha_i (one per pole) with the
    clipped matrices held fixed:

        min_{alpha >= 0} sum_{c,n} (B[c,n] - sum_i alpha_i Phi[n,i] Cv[c,i])^2

    Every term of the resulting model is PSD (alpha_i >= 0, C_pos_i PSD,
    Phi >= 0), so the whole fit is PSD at every energy.

    Returns (alpha, C_scaled) with C_scaled[i] = alpha_i * C_pos[i].

    FAST EXACT PATH (profiled 2026-10-09: the full-design nnls was 53%
    of the production gram fit). The design A[(c,n), i] = Cv[c,i] *
    Phi[n,i] is never formed. Its normal equations FACTORIZE exactly:

        G[i,j] = sum_{c,n} Cv[c,i]Cv[c,j] Phi[n,i]Phi[n,j]
               = (Cv^T Cv)[i,j] * (Phi^T Phi)[i,j]   (Hadamard product)
        h[i]   = sum_c Cv[c,i] * (B @ Phi)[c,i]

    and with G = L L^T (Cholesky),
        ||A a - b||^2 = ||L^T a - L^{-1} h||^2 + const,
    so nnls on the (M x M) triangular system has EXACTLY the same
    minimizer as nnls on the (n_ch*N, M) design. Cost O((N + n_ch)*M^2)
    with NO (n_ch*N, M) materialization (measured on Ag3Au3: 729*451
    rows, up to 260 MB per call, 0.67 s per Lawson-Hanson solve, ~84
    calls per fit). The solution is KKT-checked (gradient of the
    true quadratic); anything suspicious falls back to the original
    full-design nnls, so the fast path can only change the SPEED.
    """
    Phi = np.asarray(Phi)
    M = Phi.shape[1]
    n_ch, N = B.shape
    Cv = _channel_values(C_pos, chans)

    alpha = _refit_scales_fast(Phi, B, Cv)
    if alpha is None:
        # design: rows = (channel, sample), cols = poles
        A = (Cv[:, None, :] * Phi[None, :, :]).reshape(n_ch * N, M)
        b = np.asarray(B).reshape(-1)
        alpha, _ = nnls(A, b)
    return alpha, C_pos * alpha[:, None, None]


def _refit_scales_fast(Phi, B, Cv, kkt_tol=1e-6):
    """Factored normal-equations NNLS for refit_scales.

    Returns alpha, or None when the fast path is untrustworthy
    (Cholesky failure on a rank-deficient pole set, non-finite
    solution, or KKT violation) - the caller then runs the original
    full-design nnls. The KKT check uses the TRUE objective gradient
    w = h - G alpha of the quadratic (identical to the full design's
    gradient in exact arithmetic): on the support w ~ 0, off the
    support w <= 0.
    """
    try:
        P2 = Phi.T @ Phi                       # (M, M)
        C2 = Cv.T @ Cv                         # (M, M)
        G = P2 * C2                            # exact Hadamard factorization
        G = 0.5 * (G + G.T)
        BP = np.asarray(B) @ Phi               # (n_ch, M)
        h = np.einsum("cm,cm->m", Cv, BP)      # (M,)
        L = np.linalg.cholesky(G)
        y = solve_triangular(L, h, lower=True, check_finite=False)
        alpha, _ = nnls(L.T, y)                # (M x M) system
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(alpha)):
        return None
    # KKT check on the true quadratic
    w = h - G @ alpha
    scale = max(float(np.abs(h).max()), 1e-300)
    support = alpha > 0
    viol = 0.0
    if support.any():
        viol = max(viol, float(np.abs(w[support]).max()))
    if (~support).any():
        viol = max(viol, float(w[~support].max()))
    if viol > kkt_tol * scale:
        return None
    return alpha


def psd_solve_admm(Phi, B, chans, p, iters=200, rho=None, warm=None,
                   verbose=False):
    """PSD-constrained coefficient solve at fixed poles.

        min_{C_i PSD}  || Phi @ X - B^T ||_F^2

    solved by ADMM: the X-update is a single (M x M) linear solve per
    iteration (the unconstrained LS system plus a proximal ridge toward
    the projected variable), the Z-update is the per-pole PSD projection.
    Convex in X, so this finds the PSD-OPTIMAL residues at the given
    poles - the clip + scale refit is only used as the warm start.

    Returns (C, X): (M, p, p) PSD residues and the packed (M, n_ch).
    """

    A = np.asarray(Phi)
    Y = np.asarray(B).T                       # (N, n_ch)
    M = A.shape[1]
    AtA = A.T @ A
    rho = max(float(np.trace(AtA)) / M, 1e-12)
    AtA_r = AtA + rho * np.eye(M)
    AtY = A.T @ Y                             # (M, n_ch)

    # AtA_r is fixed for all iterations: factor once (guide 4.7)
    from scipy.linalg import cho_factor, cho_solve
    cF = cho_factor(AtA_r)

    if warm is not None:
        X = np.asarray(warm).copy()
    else:
        X = cho_solve(cF, AtY)
    ia, ib, re = channel_indices(chans)
    C = unpack_coefficients(X, chans, p)
    Z = project_psd(C)[0]
    U = np.zeros_like(Z)
    # over-relaxation (guide 2.3): alpha in (0, 2) keeps ADMM convergent
    # and typically cuts the iteration count several-fold at the same
    # solution quality.
    alpha = 1.6
    # fused-iteration workspaces, allocated ONCE and reused: the
    # X-update RHS (G), the over-relaxed candidate (C_hat) and the
    # projection input (S). The old loop's per-iteration temporaries
    # - (Z - U), the fancy-index copy, np.where, AtY + rho*G, C_hat,
    # C_hat + U, U + C_hat - Z_new - are all gone (numba path; the
    # fallback reproduces the old arithmetic exactly).
    G = np.empty_like(AtY)
    C_hat = np.empty_like(Z)
    S = np.empty_like(Z)
    for it in range(iters):
        # X-update: min ||A X - Y||^2 + rho ||X - (Z - U)||^2
        if _njit is not None:
            _admm_ghat_kernel(Z, U, ia, ib, re, AtY, rho, G)
        else:
            dvals = (Z - U)[:, ia, ib]          # (M, n_ch), vectorized
            Gs = np.where(re[None, :], dvals.real, dvals.imag)
            np.multiply(Gs, rho, out=Gs)
            np.add(AtY, Gs, out=G)
        X = cho_solve(cF, G)
        C = unpack_coefficients(X, chans, p)
        if _njit is not None:
            _admm_c_hat_kernel(C, Z, alpha, C_hat)
        else:
            np.multiply(C, alpha, out=C_hat)
            np.add(C_hat, np.multiply(Z, 1.0 - alpha), out=C_hat)
        np.add(C_hat, U, out=S)
        Z_new = project_psd(S)[0]
        # U = U + C_hat - Z_new, in place (same associativity as the
        # original expression - bitwise identical)
        np.add(U, C_hat, out=U)
        np.subtract(U, Z_new, out=U)
        primal = float(np.linalg.norm(C - Z_new))
        if primal < 1e-6 * max(1.0, float(np.linalg.norm(Z_new))):
            Z = Z_new
            break
        if verbose and it % 50 == 0:
            print(f"[admm] it {it}: primal {primal:.3g}")
        Z = Z_new
    return Z, unpack_packed(Z, chans)


def unpack_packed(Z, chans):
    """PSD residues (M, p, p) -> packed (M, n_ch) real channel values
    (vectorized)."""
    ia, ib, re = channel_indices(chans)
    vals = Z[:, ia, ib]
    return np.where(re[None, :], vals.real, vals.imag)


def psd_refine(E, B, centers, widths, C, chans, admm_iters=200):
    """PSD enforcement: clip + scale refit warm start, then the ADMM
    PSD-constrained solve at the fixed poles.

    Returns dict with C (PSD residue matrices), alpha, n_clipped,
    worst_eig, and unweighted residual R (n_ch, N) of the refined model.
    """
    C_pos, n_clipped, worst = project_psd(C)
    Phi = lorentzian_basis(E, centers, widths)
    alpha, C_scaled = refit_scales(Phi, B, C_pos, chans)

    p = C.shape[1]
    # pack the warm start (M, n_ch)
    X_warm = unpack_packed(C_scaled, chans)
    C_admm, X_admm = psd_solve_admm(Phi, B, chans, p, iters=admm_iters,
                                    warm=X_warm)

    # keep whichever model fits better
    def rms_of_C(C_model):
        G = _channel_values(C_model, chans)          # (n_ch, M)
        R = np.asarray(B) - G @ Phi.T
        return float(np.sqrt(np.mean(R ** 2))), R

    r_scaled, R_scaled = rms_of_C(C_scaled)
    r_admm, R_admm = rms_of_C(C_admm)
    if r_admm <= r_scaled:
        return dict(C=C_admm, alpha=alpha, n_clipped=n_clipped,
                    worst_eig=worst, R=R_admm)
    return dict(C=C_scaled, alpha=alpha, n_clipped=n_clipped,
                worst_eig=worst, R=R_scaled)


# ==========================================================================
# SECTION: validate
# ==========================================================================
"""Validation: held-out error, PSD minimum eigenvalue on a dense grid,
complex-plane decay check, and the dispersive-basis diagnostic.

Hermiticity of the fit is exact by construction; the checks here target
the properties that are NOT automatic: PSD between samples, uniform decay
in the complex plane (hidden cancelling pairs), and overfitting.
"""

import numpy as np


def evaluate(E_grid, centers, widths, C):
    """Model on a real energy grid: (n, p, p) complex Hermitian."""
    Phi = lorentzian_basis(E_grid, centers, widths)
    return basis_apply(Phi, C)


def frobenius_rms_error(F_fit, F_ref):
    """sqrt(sum ||F_fit_n - F_ref_n||_F^2 / sum ||F_ref_n||_F^2)."""
    num = float(np.sum(np.abs(F_fit - F_ref) ** 2))
    den = float(np.sum(np.abs(F_ref) ** 2))
    return np.sqrt(num / max(den, 1e-300))


def psd_min_eigenvalue(E_grid, centers, widths, C):
    """Minimum over the grid of the smallest eigenvalue of the model."""
    F_fit = evaluate(E_grid, centers, widths, C)
    return float(np.min(np.linalg.eigvalsh(F_fit)))


def complex_plane_limsup(centers, widths, C, radius_factor=10.0,
                         n_theta=181, r_min_factor=1.5):
    """Max ||F_fit(z)|| on large semicircles in the upper half plane.

    A fit that looks fine on the real axis but is large in the complex
    plane signals hidden cancellation between near-duplicate pairs.
    Returns (max_norm, ratio_to_data_max, argmax_z) where the ratio uses
    the on-axis maximum of the model as the reference scale.
    """
    E_mid = 0.5 * (float(np.min(centers)) + float(np.max(centers)))
    r = radius_factor * max(float(np.max(centers)) - E_mid,
                            float(np.min(centers)) - E_mid, 1.0) + r_min_factor
    theta = np.linspace(0.0, np.pi, n_theta)
    z = E_mid + r * np.exp(1j * theta)
    Phi = (widths ** 2)[None, :] / ((z[:, None] - centers[None, :]) ** 2 +
                                    (widths ** 2)[None, :])
    Fz = basis_apply(Phi, C)
    norms = np.sqrt(np.sum(np.abs(Fz) ** 2, axis=(1, 2)))
    on_axis = evaluate(np.linspace(E_mid - r, E_mid + r, 501),
                       centers, widths, C)
    ref = float(np.max(np.sqrt(np.sum(np.abs(on_axis) ** 2, axis=(1, 2)))))
    i = int(np.argmax(norms))
    return float(norms[i]), float(norms[i] / max(ref, 1e-300)), complex(z[i])


def held_out_error(E, F, centers, widths, C, test_idx, chans=None):
    """Error on held-out samples with coefficients retrained on the rest.

    Poles are kept from the full fit (mild leakage); only the linear
    coefficient solve is redone on the training subset, which is where
    most of the freedom is. Returns Frobenius rms error on the test set.
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    p = F.shape[1]
    chans = hermitian_channels(p) if chans is None else chans
    mask = np.ones(E.size, dtype=bool)
    mask[np.asarray(test_idx)] = False
    B = pack_responses(F, chans)
    Phi_tr = lorentzian_basis(E[mask], centers, widths)
    X, _ = solve_coefficients(Phi_tr, B[:, mask])
    C_tr = unpack_coefficients(X, chans, p)
    F_test_fit = evaluate(E[~mask], centers, widths, C_tr)
    return frobenius_rms_error(F_test_fit, F[~mask])


def dispersive_diagnostic(E, F, centers, widths, chans=None):
    """Caveat-1 test: refit with the augmented (absorptive + dispersive)
    basis and report the relative size of the dispersive coefficients.

    Large dispersive components mean the restricted absorptive-only basis
    is wrong for the data (background/asymmetric tails present).
    Returns (ratio, n_abs_coeff) where ratio = ||disp|| / (||abs|| + eps).
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    p = F.shape[1]
    chans = hermitian_channels(p) if chans is None else chans
    B = pack_responses(F, chans)
    d = E[:, None] - centers[None, :]
    w2 = (widths ** 2)[None, :]
    Phi_abs = w2 / (d * d + w2)
    Phi_dis = (d * widths[None, :]) / (d * d + w2)
    A = np.concatenate([Phi_abs, Phi_dis], axis=1)
    X, _ = solve_coefficients(A, B)
    n_abs = Phi_abs.shape[1]
    num = float(np.sqrt(np.sum(X[n_abs:] ** 2)))
    den = float(np.sqrt(np.sum(X[:n_abs] ** 2)))
    return num / max(den, 1e-300), n_abs


# ==========================================================================
# SECTION: sampling
# ==========================================================================
"""Adaptive refinement of the energy sampling grid for level-width
functions, designed to run alongside the shared-pole Lorentzian fit.

Two operations on the current samples (E, F(E)) of the Hermitian matrix
function:

- suggest_insertions: energies where the sampling under-resolves the
  function. Criterion: per-interval interpolation discrepancy - the
  difference between the piecewise-linear and the local-cubic
  (4-point Lagrange) interpolant at the interval midpoint, relative to
  the local function scale. Split such intervals at the midpoint.
  Optional extra sources: fit-residual peaks and barely-resolved fit
  poles (a pole width comparable to the local spacing may be a genuine
  sharp resonance that finer sampling should confirm).

- suggest_removals: interior samples that carry no information - the
  leave-one-out interpolant (4-point Lagrange over the neighbors)
  reproduces them to within a small tolerance of the local scale, and
  removing them does not open a gap wider than max_gap_factor times
  the median spacing.

Both criteria are purely local and work on arbitrary (non-uniform)
grids. Typical use, one refinement round:

    E_new = np.sort(np.union1d(E[keep], insert))
    # evaluate F at insert (TBtrans / self-energy calculator), then refit

The Zandpack wiring (contour -> TBtrans -> read_data) lives in the
bench/example scripts; this module is numpy-only.
"""

import numpy as np


def _frobenius_scale(F):
    """Per-sample Frobenius norms and the global scale."""
    Fn = np.linalg.norm(F, axis=(1, 2))
    scale = float(Fn.max()) if Fn.size else 1.0
    return Fn, max(scale, 1e-300)


def _lagrange_weights(nodes, x):
    """Lagrange basis polynomials through `nodes`, evaluated at scalar x:
    y(x) = sum_k w_k * y_k with sum_k w_k == 1."""
    nodes = np.asarray(nodes, dtype=float)
    K = nodes.size
    w = np.empty(K)
    for k in range(K):
        others = np.delete(nodes, k)
        w[k] = np.prod(x - others) / np.prod(nodes[k] - others)
    return w


def interpolation_discrepancy(E, F):
    """Per-interval under-resolution estimate.

    For each interval [E_i, E_{i+1}]: the difference between the local
    cubic interpolant (4-point Lagrange over the nearest samples) and
    the linear interpolant, both evaluated at the midpoint, maximized
    over all matrix entries. Raw (unnormalized) units.

    Returns (disc, local_scale), each of length N-1: the discrepancy
    and the local function scale max(||F_i||, ||F_{i+1}||).
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    N = E.size
    disc = np.zeros(N - 1)
    for i in range(N - 1):
        mid = 0.5 * (E[i] + E[i + 1])
        lo = max(i - 1, 0)
        hi = min(i + 2, N - 1)
        nodes = E[lo:hi + 1]
        # ensure 4 distinct nodes where available
        if hi - lo + 1 < 4:
            if lo == 0:
                hi = min(lo + 3, N - 1)
            else:
                lo = max(hi - 3, 0)
            nodes = E[lo:hi + 1]
        w = _lagrange_weights(nodes, mid)
        cubic = np.tensordot(w, F[lo:hi + 1], axes=(0, 0))
        lin = 0.5 * (F[i] + F[i + 1])
        disc[i] = float(np.max(np.abs(cubic - lin)))
    Fn, _ = _frobenius_scale(F)
    local = np.maximum(Fn[:-1], Fn[1:])
    return disc, local


def suggest_insertions(E, F, tol=5e-3, max_add=None, floor_frac=1e-3,
                       residual=None, poles=None, pole_factor=2.0):
    """Propose new sample energies where the grid under-resolves F.

    Parameters
    ----------
    E, F : current samples, (N,) and (N, p, p)
    tol : flag an interval when its midpoint interpolation discrepancy
        exceeds tol * max(local scale, floor_frac * global scale)
    max_add : cap on the number of proposed points (None = no cap)
    residual : optional fit residuals (n_ch, N) or per-sample norms (N,);
        the worst residual peaks also get a new point each (the fit
        struggling at a sample means the sampling is too coarse there)
    poles : optional fit poles (e, w); poles with w < pole_factor * the
        local spacing are bracketed with two extra points each
    floor_frac : local-scale floor as a fraction of the global scale

    Returns a sorted array of proposed energies (not yet in E).
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    N = E.size
    _, scale = _frobenius_scale(F)
    floor = floor_frac * scale

    out = []
    if N >= 3:
        disc, local = interpolation_discrepancy(E, F)
        flag = disc > tol * np.maximum(local, floor)
        idx = np.where(flag)[0]
        # worst intervals first when capping
        if max_add is not None and idx.size > max_add:
            idx = idx[np.argsort(-disc[idx])][:max_add]
        out.extend(0.5 * (E[idx] + E[idx + 1]))

    if residual is not None:
        r = np.asarray(residual)
        rn = np.sqrt(np.sum(r ** 2, axis=0)) if r.ndim == 2 else np.abs(r)
        if rn.size and rn.max() > 0:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rn, prominence=0.1 * rn.max())
            if peaks.size == 0:
                peaks = np.array([int(np.argmax(rn))])
            for i in sorted(peaks, key=lambda j: -rn[j]):
                if max_add is not None and len(out) >= max_add:
                    break
                out.append(E[i])

    if poles is not None:
        e_p = np.asarray(poles[0], dtype=float)
        w_p = np.asarray(poles[1], dtype=float)
        if e_p.size:
            d_near = distance_to_nearest_sample(E, e_p)
            for ei, wi, di in zip(e_p, w_p, d_near):
                if wi < pole_factor * di and E.min() < ei < E.max():
                    if max_add is not None and len(out) >= max_add:
                        break
                    out.extend([ei - wi, ei + wi])

    out = np.array(sorted(set(np.asarray(out, dtype=float))))
    out = out[(out > E.min()) & (out < E.max())]
    return out


def leave_one_out_error(E, F):
    """Per-sample redundancy estimate for interior samples.

    For each interior i: the 4-point Lagrange interpolant through the
    nearest other samples, evaluated at E_i, compared with F_i. The
    error is the max over matrix entries. Edge samples are excluded
    (error = inf there; they are never removable).

    Returns (err, local_scale), each of length N.
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    N = E.size
    err = np.full(N, np.inf)
    for i in range(2, N - 2):
        nodes_idx = np.array([i - 2, i - 1, i + 1, i + 2])
        w = _lagrange_weights(E[nodes_idx], E[i])
        est = np.tensordot(w, F[nodes_idx], axes=(0, 0))
        err[i] = float(np.max(np.abs(est - F[i])))
    Fn, _ = _frobenius_scale(F)
    return err, Fn


def suggest_insertions_derivative(E, F, tol=3.0, max_add=48):
    """Fit-free insertion proposals from the derivative of the
    abs-summed function s(E) = sum_ij |F_ij(E)|.

    Intervals are scored by max|ds/dE| over their two endpoints;
    midpoints of the steepest intervals are proposed first until the
    score drops below tol * (median |ds/dE|) or max_add is reached.
    Steep flanks - band-edge onsets, notch walls, resonance shoulders -
    are exactly where a sampling grid needs density, and this criterion
    needs NO fit at all (the fit-based suggest_insertions requires a
    fit per round; this one is a cheap pre-pass).

    tol : score threshold in units of the median |ds/dE| (robust
        scale: flat regions dominate the median, so tol ~ 3 flags the
        genuinely steep parts).
    Returns a sorted array of proposed energies (interval midpoints).
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    order = np.argsort(E)
    E, F = E[order], F[order]
    if E.size < 3:
        return np.zeros(0)

    s = np.sum(np.abs(F), axis=tuple(range(1, F.ndim)))
    ds = np.abs(np.gradient(s, E))
    scale = float(np.median(ds))
    if scale <= 0:
        return np.zeros(0)

    score = np.maximum(ds[:-1], ds[1:])          # per-interval steepness
    cand = np.where(score > tol * scale)[0]
    if cand.size == 0:
        return np.zeros(0)
    cand = cand[np.argsort(-score[cand])]        # steepest first
    if cand.size > max_add:
        cand = cand[:max_add]
    mids = 0.5 * (E[cand] + E[cand + 1])
    return np.unique(mids)


def suggest_removals(E, F, tol=1e-2, floor_frac=1e-3, max_gap_factor=3.0):
    """Interior samples that can be dropped without information loss.

    A sample is redundant when its leave-one-out interpolation error is
    below tol * max(||F_i||, floor_frac * global scale) AND dropping it
    (together with any already-accepted drop that merges into the same
    gap) keeps the resulting gap below max_gap_factor * the median
    spacing. Edge samples are never dropped.

    Returns a sorted index array into E.
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    N = E.size
    if N < 5:
        return np.array([], dtype=int)
    err, Fn = leave_one_out_error(E, F)
    _, scale = _frobenius_scale(F)
    floor = floor_frac * scale
    redundant = err <= tol * np.maximum(Fn, floor)

    d = np.diff(E)
    max_gap = max_gap_factor * float(np.median(d))
    keep = np.ones(N, dtype=bool)
    remove = []
    for i in range(2, N - 2):
        if not redundant[i]:
            continue
        # gap that would result: previous kept sample to next kept sample
        prev = i - 1
        while not keep[prev]:
            prev -= 1
        nxt = i + 1
        while not keep[nxt]:
            nxt += 1
        if E[nxt] - E[prev] <= max_gap:
            keep[i] = False
            remove.append(i)
    return np.array(sorted(remove), dtype=int)


def refine_grid(E, F, insert_tol=5e-3, remove_tol=1e-2, max_add=None,
                **kwargs):
    """One refinement round: insertion + removal proposal.

    Returns dict with:
      insert : sorted new energies to sample (to be evaluated externally)
      remove_idx : indices into E that are redundant
      keep_idx : indices of the surviving old samples
    """
    E = np.asarray(E, dtype=float)
    insert = suggest_insertions(E, F, tol=insert_tol, max_add=max_add,
                                **kwargs)
    remove_idx = suggest_removals(E, F, tol=remove_tol)
    keep = np.setdiff1d(np.arange(E.size), remove_idx)
    return dict(insert=insert, remove_idx=remove_idx, keep_idx=keep)


# ==========================================================================
# SECTION: pipeline
# ==========================================================================
"""Shared-pole Lorentzian fit of Hermitian matrix-valued functions.

    F(E) ~= sum_i C_i * w_i^2 / ((E - e_i)^2 + w_i^2),   C_i = C_i^*

with shared pole pairs (e_i, w_i) across all matrix entries, Hermitian
(optionally PSD) residue matrices, and strictly proper decay (no offset
term; the basis decays identically).

Pipeline (implementation plan):
    prepare -> pole init (skrf VectorFitting or linspace)
            -> variable projection (linear LS for C_i inside a bounded
               nonlinear fit over e_i, w_i)
            -> prune negligible/duplicate pairs, add poles at residual
               peaks until the target error is met
            -> optional PSD refinement (eigenvalue clip + scale refit)
            -> validation numbers
"""

from dataclasses import dataclass, field
from typing import Optional

import sys

import numpy as np


__all__ = ["fit", "FitResult", "prepare", "FitData"]


@dataclass
class FitResult:
    """Fit outcome. Poles/residues are stored in RAW units (as given in),
    with the scaled internal copies kept for evaluation helpers."""

    e: np.ndarray              # (M,) raw pole centers
    w: np.ndarray              # (M,) raw widths (same unit as e)
    C: np.ndarray              # (M, p, p) raw Hermitian residue matrices
    p: int
    rms: float                 # Frobenius rms error over the samples (raw)
    data: FitData        # scaled data + unit conversions
    e_scaled: np.ndarray
    w_scaled: np.ndarray
    C_scaled: np.ndarray
    n_poles: int
    chans: list = field(default_factory=list)  # channels used in the solve
    history: dict = field(default_factory=dict)
    psd: Optional[dict] = None

    def evaluate(self, E_raw):
        """Model on raw energies, raw units: (n, p, p)."""
        return self.data.evaluate(self.e_scaled, self.w_scaled,
                                  self.C_scaled, E_raw)


def _quad_widths(E):
    """Trapezoid measure around each sample (review 4.1).

    Refined grids cluster samples at peaks; an unweighted per-sample
    cost then over-weights dense clusters and leaves sparse regions
    unconstrained. The quadrature width makes the cost approximate
    an integral over energy. Near-duplicate samples are clipped to a
    quarter of the median spacing so they cannot vanish.
    """
    E = np.sort(np.asarray(E, dtype=float))
    if E.size < 2:
        return np.ones(1)
    d = np.diff(E)
    w = np.empty(E.size)
    w[0] = d[0]
    w[-1] = d[-1]
    if E.size > 2:
        w[1:-1] = 0.5 * (d[:-1] + d[1:])
    pos = d[d > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    return np.maximum(w, 0.25 * med)


def _quad_rms(F_fit, F_s, qw):
    """Quadrature-weighted Frobenius rms (review 4.1): approximates the
    L2 error over the energy window instead of over sample count."""
    d = np.asarray(F_fit) - np.asarray(F_s)
    num = float(np.sum(qw * np.sum(np.abs(d) ** 2, axis=(1, 2))))
    return float(np.sqrt(num / max(float(qw.sum()), 1e-300)))


def _sample_weights(F, mode, eps_frac=1e-6, quad=None):
    """Cost weights for the LS (review 4.1, with 4.4 resolved).

    'relative' (default): per-sample weight 1 / max(||F_n||, eps_floor)
    (half-relative cost; kept deliberately - switching to the fully
    relative 1/||F_n||^2 was tried per the review and measurably
    over-emphasized weak samples, degrading peak fidelity on both the
    recovery tests and the level-width examples, which span ~4 orders
    of magnitude). 'quad' (optional trapezoid widths per sample)
    multiplies the weights so the cost approximates an energy integral
    - refined and uniform grids then behave consistently (review 4.1).
    """
    norms = np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2)))
    if mode == "absolute":
        return None if quad is None else np.asarray(quad, dtype=float)
    if mode != "relative":
        raise ValueError(f"unknown weight mode {mode!r}")
    floor = eps_frac * max(float(norms.max()), 1e-300)
    w = 1.0 / np.maximum(norms, floor)
    if quad is not None:
        w = w * np.asarray(quad, dtype=float)
    return w


def fit(E, F, n_poles: Optional[int] = None, n_init: Optional[int] = None,
        target_rms: float = 5e-3, max_poles: Optional[int] = None,
        weight: str = "relative", init: str = "skrf", psd: bool = False,
        prune_rel_tol: float = 1e-3, verbose: bool = False,
        max_nfev: Optional[int] = None, rcond=None,
        max_nfev_polish: int = 1500, channel_tol: float = 0.0,
        e_margin: float = 0.15, tail_anchors: bool = True,
        anchor_weight: float = 0.05, ridge: float = 0.05,
        w_min: Optional[float] = None,
        local_floor_frac: float = 1.0,
        psd_force: bool = False,
        eta_raw: Optional[float] = None,
        eta_floor_mult: float = 1.0,
        w_max_raw: Optional[float] = None,
        init_poles=None,
        operator_threshold: int = 3_000_000,
        tr_solver="lsmr", gram_cut: float = 1e-9) -> FitResult:
    """Fit shared-pole Lorentzians to samples of a Hermitian matrix function.

    The width floor is chosen in two stages: first the STABLE tight floor
    (the median sample spacing - a Lorentzian with half-width equal to
    half the spacing has about one sample across its FWHM and rings
    between samples, so 1.0 x spacing is the minimum safe floor; pass
    eta_raw to additionally forbid poles narrower than the imaginary
    broadening of the sampled data), and if the fit is still clearly
    inadequate (rms > 3 * target_rms and > 3e-3), the data genuinely
    contains sharp resolvable structure and the fit is redone with the
    loose absolute floor (1e-3 in scaled units). The better fit is
    returned. Pass w_min explicitly to pin the floor yourself.

    See _fit_impl for the remaining parameters.
    """
    E = np.asarray(E, dtype=float).ravel()
    if w_min is not None:
        res = _fit_impl(E, F, n_poles=n_poles, n_init=n_init,
                        target_rms=target_rms, max_poles=max_poles,
                        weight=weight, init=init, psd=psd,
                        prune_rel_tol=prune_rel_tol, verbose=verbose,
                        max_nfev=max_nfev, rcond=rcond,
                        max_nfev_polish=max_nfev_polish,
                        channel_tol=channel_tol, e_margin=e_margin,
                        tail_anchors=tail_anchors, anchor_weight=anchor_weight,
                        ridge=ridge, w_min=w_min,
                        local_floor_frac=local_floor_frac,
                        psd_force=psd_force, eta_raw=eta_raw,
                        eta_floor_mult=eta_floor_mult,
                        w_max_raw=w_max_raw, init_poles=init_poles,
                        operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut)
        res.history["w_min"] = float(w_min)
        return res

    # two-stage floor (scaled units, window mapped to [-1, 1])
    span = max(float(E.max() - E.min()), 1e-12)
    e_scale = 0.5 * span
    dE = float(np.median(np.diff(np.sort(E)))) if E.size > 1 else span
    tight = max(1e-3, 1.0 * dE / e_scale)
    if eta_raw is not None:
        # structure narrower than the imaginary broadening of the
        # sampled data cannot exist in the data (review 4.3);
        # eta_floor_mult > 1 additionally smooths poles at the
        # resolution limit, which otherwise make the fit oscillate
        # around the sampling function between samples
        tight = max(tight, eta_floor_mult * float(eta_raw) / e_scale)
    loose = 1e-3
    if eta_raw is not None:
        # the eta floor (with the same multiplier) applies to BOTH
        # stages: the data's own broadening is a hard physical limit
        loose = max(loose, eta_floor_mult * float(eta_raw) / e_scale)

    kw = dict(n_poles=n_poles, n_init=n_init, target_rms=target_rms,
              max_poles=max_poles, weight=weight, init=init, psd=psd,
              prune_rel_tol=prune_rel_tol, verbose=verbose,
              max_nfev=max_nfev, rcond=rcond,
              max_nfev_polish=max_nfev_polish, channel_tol=channel_tol,
              e_margin=e_margin, tail_anchors=tail_anchors,
              anchor_weight=anchor_weight, ridge=ridge,
              local_floor_frac=local_floor_frac, psd_force=psd_force,
              eta_raw=eta_raw, eta_floor_mult=eta_floor_mult,
              w_max_raw=w_max_raw, init_poles=init_poles,
              operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut)

    res1 = _fit_impl(E, F, w_min=tight, **kw)
    res1.history["w_min"] = tight
    if res1.rms <= max(3.0 * target_rms, 3e-3):
        if verbose:
            print(f"[fit] two-stage floor: tight w_min={tight:.4g} "
                  f"sufficient (rms={res1.rms:.4g})")
        return res1
    if verbose:
        print(f"[fit] tight floor (w_min={tight:.4g}) insufficient "
              f"(rms={res1.rms:.4g}); retrying with loose floor")
    res2 = _fit_impl(E, F, w_min=loose, **kw)
    res2.history["w_min"] = loose
    if verbose:
        print(f"[fit] two-stage floor: keeping the "
              f"{'loose' if res2.rms < res1.rms else 'tight'} fit "
              f"(rms {res2.rms:.4g} vs {res1.rms:.4g})")
    return res2 if res2.rms < res1.rms else res1


def fit_fixed_poles(E, F, centers, widths, psd=True, psd_force=True,
                   weight="relative", ridge=0.0, rcond=None,
                   tail_anchors=True, anchor_weight=0.05,
                   admm_iters=200, target_rms=5e-3, verbose=False):
    """Fit ONLY the residue coefficients at FIXED poles - no pole
    initialization, no relocation, no adaptive loop.

    Pole inheritance for symmetric systems: fit ONE electrode with
    the full machinery (fit / fit_boosted), then hand the OTHER
    electrode the fitted (centers, widths) and solve just its residue
    matrices. The symmetric partner shares the spectral structure, so
    the inherited pole set should be near-optimal for it too - at the
    cost of ONE coefficient solve (seconds) instead of a full
    adaptive fit.

    The coefficient path is exactly the final solve of _fit_impl:
    - weighted LS at the fixed poles (solve_coefficients, same
      sample weights / tail anchors);
    - psd=True: psd_refine (clip + scale refit warm start, then the
      convex PSD-constrained ADMM at the fixed poles), kept under the
      SAME accuracy guard as fit(): applied unconditionally when
      psd_force, else only when the PSD rms stays within
      max(3 x unconstrained, 2 x target_rms).

    RIDGE: defaults to 0.0, NOT _fit_impl's 0.05. The ridge only
    guards a solve whose poles are about to be RELOCATED (it seeds
    TRF without cancellation blowups); here the poles stay put, and
    a Tikhonov penalty measurably BIASES the re-solve whenever the
    inherited poles overlap and need large cancelling coefficients
    (measured on Ag3Au3, 100 poles: x24 rms degradation at ridge
    0.05 vs x0.97 at ridge 0 - validation round 2026-10-09).

    centers/widths are RAW units (e.g. res.e, res.w of another
    FitResult). Pass poles that are already valid for the data window;
    nothing clamps or widens them here.

    Returns a FitResult (poles returned unchanged; history carries
    fixed_poles=True so callers can tell it apart from a full fit).
    """
    E = np.asarray(E, dtype=float).ravel()
    fd = prepare(E, F)
    E_s, F_s = fd.E, fd.F
    p = F_s.shape[1]
    chans = hermitian_channels(p)
    B = pack_responses(F_s, chans)
    e_s = fd.scale_centers(np.asarray(centers, dtype=float))
    w_s = np.asarray(widths, dtype=float) / fd.e_scale

    qw = _quad_widths(E_s)
    sw = _sample_weights(F_s, weight, quad=qw)
    N_real = E_s.size
    E_fit, B_fit, sw_fit = E_s, B, sw
    if tail_anchors:
        span = float(E_s.max() - E_s.min())
        d = np.array([0.05, 0.15, 0.35, 0.65]) * 0.5 * span
        E_anchor = np.concatenate([E_s.min() - d, E_s.max() + d])
        B_anchor = np.zeros((B.shape[0], E_anchor.size))
        ref_w = 1.0 if sw is None else float(np.median(sw))
        w_anchor = anchor_weight * ref_w
        E_fit = np.concatenate([E_s, E_anchor])
        B_fit = np.concatenate([B, B_anchor], axis=1)
        if sw is None:
            sw_fit = np.concatenate(
                [np.ones(N_real), np.full(E_anchor.size, anchor_weight)])
        else:
            sw_fit = np.concatenate(
                [sw, np.full(E_anchor.size, w_anchor)])

    Phi = lorentzian_basis(E_fit, e_s, w_s)
    X, _ = solve_coefficients(Phi, B_fit, sample_weights=sw_fit,
                              rcond=rcond, ridge=ridge)
    C_s = unpack_coefficients(X, chans, p)
    Phi_real = lorentzian_basis(E_s, e_s, w_s)
    rms = _quad_rms(basis_apply(Phi_real, C_s), F_s, qw)
    history = dict(fixed_poles=True, n_poles=int(e_s.size),
                   rms=float(rms), n_channels=len(chans))
    if verbose:
        print(f"[fit_fixed_poles] {e_s.size} poles, "
              f"unconstrained rms {rms:.4e}")

    psd_info = None
    if psd:
        refined = psd_refine(E_fit, B_fit, e_s, w_s, C_s, chans,
                             admm_iters=admm_iters)
        refined["rms"] = _quad_rms(
            basis_apply(Phi_real, refined["C"]), F_s, qw)
        refined["applied"] = (
            psd_force or
            refined["rms"] <= max(rms * 3.0, target_rms * 2.0))
        history["psd_applied"] = bool(refined["applied"])
        if refined["applied"]:
            rms_unc = rms
            C_s = refined["C"]
            rms = refined["rms"]
            history["rms_psd"] = rms
        else:
            rms_unc = rms
        psd_info = refined
        if verbose:
            print(f"[fit_fixed_poles] psd rms {refined['rms']:.4e} vs "
                  f"unconstrained {rms_unc:.4e} -> "
                  f"{'APPLIED' if refined['applied'] else 'rejected'}"
                  f"{' (forced)' if psd_force else ''}")

    return FitResult(
        e=fd.unscale_centers(e_s),
        w=fd.unscale_widths(w_s),
        C=fd.unscale_coeffs(C_s),
        p=p,
        rms=float(rms),
        data=fd,
        e_scaled=e_s,
        w_scaled=w_s,
        C_scaled=C_s,
        n_poles=int(e_s.size),
        chans=chans,
        history=history,
        psd=psd_info,
    )


def _fit_impl(E, F, n_poles: Optional[int] = None, n_init: Optional[int] = None,
        target_rms: float = 5e-3, max_poles: Optional[int] = None,
        weight: str = "relative", init: str = "skrf", psd: bool = False,
        prune_rel_tol: float = 1e-3, verbose: bool = False,
        max_nfev: Optional[int] = None, rcond=None,
        max_nfev_polish: int = 1500, channel_tol: float = 0.0,
        e_margin: float = 0.15, tail_anchors: bool = True,
        anchor_weight: float = 0.05, ridge: float = 0.05,
        w_min: Optional[float] = None,
        local_floor_frac: float = 1.0,
        psd_force: bool = False,
        eta_raw: Optional[float] = None,
        eta_floor_mult: float = 1.0,
        w_max_raw: Optional[float] = None,
        init_poles=None,
        operator_threshold: int = 3_000_000,
        tr_solver="lsmr", gram_cut: float = 1e-9) -> FitResult:
    """Fit shared-pole Lorentzians to samples of a Hermitian matrix function.

    Parameters
    ----------
    E : (N,) real energies (any unit)
    F : (N, p, p) complex Hermitian samples (any unit)
    n_poles : exact number of pole pairs; if None, adaptive: start from
        n_init (default 32) poles, prune, and add poles at residual peaks
        until `target_rms` or `max_poles` (default 4 * n_init) is reached.
    weight : 'relative' (per-sample 1/||F_n||, default) or 'absolute'.
    init : 'skrf' (VectorFitting pole relocation, recommended) or 'linspace'.
    init_poles : optional (e, w) arrays seeding the optimizer with an
        explicit pole set (bypasses `init`). For structure the generic
        inits cannot reach (deep narrow notches, specific basins).
    psd : enforce PSD residues after the fit (clip + nonneg scale refit).
    tr_solver : trust-region subproblem solver inside scipy's trf.
        DEFAULT 'lsmr' (since 2026-10-09): 1.5-1.8x faster for M >~ 40,
        smoother residuals, better notch fidelity — but its error at
        steep band edges is a same-sign BIAS that accumulates in
        integrated quantities (measured +2.1% band-integral on RSSE
        [3,5] vs -0.1% for 'exact'; see HANDOFF 4.11). Pass
        tr_solver='exact' for the dense-SVD path: slower, its error is a
        near-zero-mean ripple that averages out in integrals — prefer
        it when integrated currents across band edges are the target
        observable. Slower than exact below M~40 (cold small solves).
        'gram' (never_form_J_guide): the vendored trust-region loop
        never materializes J - it consumes G = J^T J / g = J^T f from
        the varproj factors (cost independent of the channel count) and
        one eigh per step. Follows the 'exact' trajectory (see the Gram
        solver section for the math); NOT bit-identical to it (weak
        directions below gram_cut are truncated - eigh resolves only
        down to ~sqrt(eps) of the largest singular value). Prefer it at
        large M*n_ch where the dense J is expensive; check the
        validation gate (validation/10_gram_solver) after scipy
        upgrades (private API, pinned 1.18.1).
    gram_cut : relative eigenvalue truncation floor of the 'gram'
        solver (default 1e-9); directions with singular value below
        gram_cut * s[0] are dropped from the trust-region model.
        MEASURED on Ag3Au3 (the system with genuinely weak directions,
        s ~ 1e-8 s[0]): 1e-7 wrongly truncates a needed direction
        (sample rms 0.0276, dense-grid min eig -1.6e5); 1e-9 restores
        it (0.0228, -3.2e-2 - same ballpark as 'exact'); 1e-11 is
        bit-identical to 1e-9 there, and on RSSE/synthetic no
        direction is ever truncated at either setting.
    channel_tol : drop Hermitian channels whose max amplitude is below
        channel_tol * (largest channel amplitude) from the optimization
        and the coefficient solve (their coefficients are set to zero).
        0.0 keeps everything. Useful at large p where the level-width
        matrix is effectively sparse.
    e_margin : how far beyond the sampling window pole centers may sit,
        as a fraction of the window span. 0.0 (default) confines centers
        to the window: a narrow pole outside the window is invisible to
        every sample but produces a huge spurious peak there, so it must
        be excluded by construction.
    tail_anchors : add low-weight zero-target pseudo-samples beyond each
        window edge (an "artificial ramp" decaying to zero) so the
        extrapolated tails are gently pulled toward zero instead of
        being left completely unconstrained.
    anchor_weight : weight of each tail anchor relative to the median
        real-sample weight.
    ridge : Tikhonov penalty on the residue matrices (scaled units).
        Cures the near-null directions of the sample LS (coincident,
        sub-resolution, or window-spanning broad poles) whose
        astronomically cancelling coefficients are the spike mechanism;
        legitimate directions (A^T A ~ O(N)) are unaffected at ~1e-5.
    w_min : lower bound on the widths (scaled units); None (default)
        uses max(1e-3, half the median sample spacing) - features narrower
        than half the sample spacing are invisible to the samples.
    local_floor_frac : post-fit cleanup for non-uniform grids: any pole
        whose width is below local_floor_frac * (distance from its center
        to the nearest sample) is widened to that local floor and the
        coefficients re-solved (accepted only behind the same rms guards
        as the duplicate merge). On a uniform grid the local floor is
        half the spacing, so this reproduces the tight-floor policy
        exactly; 0.0 disables the cleanup.

    Returns FitResult with raw-unit poles e, w and residues C.
    """
    fd = prepare(E, F)
    E_s, F_s = fd.E, fd.F
    p = F_s.shape[1]
    chans_all = hermitian_channels(p)
    B_all = pack_responses(F_s, chans_all)
    if channel_tol > 0.0:
        chmax = np.max(np.abs(B_all), axis=1)
        keep = chmax >= channel_tol * float(chmax.max())
        keep[np.argmax(chmax)] = True
    else:
        keep = np.ones(len(chans_all), dtype=bool)
    chans = [c for c, k in zip(chans_all, keep) if k]
    B = B_all[keep]
    history = {"n_channels": int(keep.sum()),
               "n_channels_dropped": int((~keep).sum())}
    # quadrature weights (review 4.1/4.4): the cost approximates an
    # energy integral, and 'relative' is a true relative error with a
    # 1e-2 eps floor
    qw = _quad_widths(E_s)
    sw = _sample_weights(F_s, weight, quad=qw)
    if w_min is None:
        dE_s = float(np.median(np.diff(np.sort(E_s)))) \
            if E_s.size > 1 else 2.0
        w_min = max(1e-3, 1.0 * dE_s)
        if eta_raw is not None:
            # eta_floor_mult > 1 smooths poles at the resolution limit
            # (they otherwise make the fit oscillate around the sampling
            # function between samples); the data itself is broadened
            # by eta, so structure below a few eta is marginal anyway
            w_min = max(w_min, eta_floor_mult * float(eta_raw) / fd.e_scale)
    # upper width bound: very broad poles (a single w ~ window-span pole
    # acts as an unconstrained constant background with a sweeping KK
    # ramp that no constant H-correction can absorb) are excluded;
    # w_max_raw is in the same (raw) units as E
    w_max = 2.5 if w_max_raw is None else float(w_max_raw) / fd.e_scale
    w_max = max(w_max, w_min * 1.001)  # keep bounds non-degenerate
    bounds_fun = default_bounds(E_s, e_margin=e_margin, w_min=w_min,
                                    w_max=w_max)

    # tail anchors: low-weight zero-target pseudo-samples beyond each edge,
    # appended AFTER the real samples (insertion suggestions only ever see
    # the real-sample part of the residual)
    N_real = E_s.size
    E_fit, B_fit, sw_fit = E_s, B, sw
    if tail_anchors:
        span = float(E_s.max() - E_s.min())
        d = np.array([0.05, 0.15, 0.35, 0.65]) * 0.5 * span
        E_anchor = np.concatenate([E_s.min() - d, E_s.max() + d])
        B_anchor = np.zeros((B.shape[0], E_anchor.size))
        ref_w = 1.0 if sw is None else float(np.median(sw))
        w_anchor = anchor_weight * ref_w
        E_fit = np.concatenate([E_s, E_anchor])
        B_fit = np.concatenate([B, B_anchor], axis=1)
        if sw is None:
            # absolute mode: anchors carry a small absolute weight; emulate
            # by giving the real samples weight 1 and anchors anchor_weight
            sw_fit = np.concatenate([np.ones(N_real),
                                     np.full(E_anchor.size, anchor_weight)])
        else:
            sw_fit = np.concatenate([sw, np.full(E_anchor.size, w_anchor)])
        history["n_tail_anchors"] = int(E_anchor.size)

    # compress once per _fit_impl (guide 2.1): every fit_poles call
    # below reuses the compressed responses
    Bc_fit = compress_channels(B_fit)

    if n_poles is not None:
        n_init = int(n_poles)
        max_poles = int(n_poles)
        adaptive = False
    else:
        n_init = int(n_init) if n_init is not None else 32
        max_poles = int(max_poles) if max_poles is not None else 4 * n_init
        adaptive = True

    # 1. pole initialization (overinitialize; pruning removes the excess).
    if verbose:
        mode = (f"fixed n_poles={n_poles}" if n_poles is not None
                else f"adaptive n_init={n_init}, max={max_poles}, "
                     f"target={target_rms:g}")
        print(f"[fit] start: {E.size} samples, p={np.asarray(F).shape[1]}, "
              f"{mode}, psd={psd}"
              + (f", force={psd_force}" if psd else "")
              + f", w_min={w_min:.4g} (scaled)")
    # init_poles=(e, w) seeds the optimization with an explicit pole set:
    # for structure the automatic relocators cannot reach from scratch
    # (e.g. a deep narrow notch carved between two humps), a seeded
    # start finds the basin the generic inits miss.
    if init_poles is not None:
        centers = np.asarray(init_poles[0], dtype=float).ravel()
        widths = np.asarray(init_poles[1], dtype=float).ravel()
        if centers.size != widths.size:
            raise ValueError("init_poles: e and w must have equal length")
        n_init = int(centers.size)
    else:
        centers, widths = _poles_init_dispatch(init, E_s, F_s, n_init,
                                               e_margin=e_margin, psd=psd,
                                               verbose=verbose)
    # clamp the initial poles strictly inside the optimizer bounds (skrf
    # relocation can return centers/widths outside them)
    lo, hi = bounds_fun(centers.size)
    eps = 1e-9
    centers = np.clip(centers, lo[0::2] + eps, hi[0::2] - eps)
    widths = np.clip(widths, lo[1::2] + eps, hi[1::2] - eps)
    history.update(init_pairs=int(centers.size))

    def rms_of(res):
        C_s = unpack_coefficients(res["X"], chans, p)
        F_fit = basis_apply(
                         lorentzian_basis(E_s, res["e"], res["w"]), C_s)
        return _quad_rms(F_fit, F_s, qw), C_s

    def psd_of(res, C_s):
        """PSD-projected model of a candidate (clip + scale refit).

        Ranking only - a REDUCED ADMM budget is enough to order
        candidates (the final psd_refine at the end of _fit_impl uses
        the full 200 iterations), and the full budget per candidate
        dominated the loop time at large p."""
        refined = psd_refine(E_fit, B_fit, res["e"], res["w"],
                                  C_s, chans, admm_iters=50)
        Phi_r = lorentzian_basis(E_s, res["e"], res["w"])
        F_psd = basis_apply( Phi_r, refined["C"])
        refined["rms"] = _quad_rms(F_psd, F_s, qw)
        return refined

    best = None

    def consider(res):
        # with psd on, candidates are tracked by the rms of their
        # PSD-PROJECTED model: the projection removes inter-pole
        # cancellation, so an excellent unconstrained candidate can be
        # poor once projected - the pipeline must optimize for what it
        # will actually return
        nonlocal best
        rms, C_s = rms_of(res)
        refined = None
        key = rms
        if psd:
            refined = psd_of(res, C_s)
            key = refined["rms"]
        if best is None or key < best["key"]:
            best = dict(rms=rms, res=res, C=C_s, key=key,
                        refined=refined)
        return key

    # 2. variable projection at the initial poles
    nfev_step = max_nfev if max_nfev is not None else 400
    # loose tolerances for the INTERMEDIATE refits (guide 2.2): they
    # only seed the next candidate, and the polish below re-refines
    # the winner tightly. The polish keeps the tight defaults.
    LOOSE = dict(ftol=1e-6, xtol=1e-6, gtol=1e-6)
    res = fit_poles(E_fit, B_fit, centers, widths, sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut,
                        bounds_fun=bounds_fun, rcond=rcond, max_nfev=nfev_step, **LOOSE)
    rms = consider(res)
    if verbose:
        print(f"[fit] init {centers.size} pairs -> rms {rms:.4e} "
              f"({res['nfev']} nfev)")

    # 3. prune + (adaptive) add loop
    for it in range(16):
        pruned = prune(best["res"]["e"], best["res"]["w"], best["C"],
                              rel_tol=prune_rel_tol)
        n_before = best["res"]["e"].size
        if pruned[0].size != n_before and pruned[0].size >= 2:
            res_p = fit_poles(E_fit, B_fit, pruned[0], pruned[1],
                                  sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut, bounds_fun=bounds_fun,
                                  rcond=rcond, max_nfev=nfev_step, **LOOSE)
            rms_p = consider(res_p)
            if verbose:
                print(f"[fit] pruned {n_before} -> {pruned[0].size} pairs, "
                      f"rms {rms_p:.4e}")
        if not adaptive:
            break
        if best["key"] <= target_rms:
            break
        if best["res"]["e"].size >= max_poles:
            break
        # in PSD mode, insert where the PROJECTED model is worst
        if psd and best["refined"] is not None:
            R_ins = best["refined"]["R"][:, :N_real]
        else:
            R_ins = best["res"]["R"][:, :N_real]
        new_c, new_w = suggest_pole_insertions(
            E_s, R_ins, best["res"]["e"],
            best["res"]["w"],
            max_insert=max(1, min(4, max_poles - best["res"]["e"].size)))
        if new_c.size == 0:
            break
        centers = np.concatenate([best["res"]["e"], new_c])
        widths = np.concatenate([best["res"]["w"], new_w])
        lo_a, hi_a = bounds_fun(centers.size)
        centers = np.clip(centers, lo_a[0::2] + 1e-9, hi_a[0::2] - 1e-9)
        widths = np.clip(widths, lo_a[1::2] + 1e-9, hi_a[1::2] - 1e-9)
        res_a = fit_poles(E_fit, B_fit, centers, widths,
                              sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut, bounds_fun=bounds_fun,
                              rcond=rcond, max_nfev=nfev_step, **LOOSE)
        rms_a = consider(res_a)
        if verbose:
            print(f"[fit] added {new_c.size} pairs -> {centers.size}, "
                  f"rms {rms_a:.4e}")
        if rms_a > 0.999 * best["key"] and new_c.size == 1 and it >= 4:
            # adding poles no longer pays (key = projected rms in
            # psd mode, plain rms otherwise)
            break

    # final polish: one bounded-budget refit from the best poles, then a
    # weighted coefficient solve (real samples + tail anchors)
    res_f = fit_poles(E_fit, B_fit, best["res"]["e"], best["res"]["w"],
                          sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut, bounds_fun=bounds_fun,
                          rcond=rcond, max_nfev=max_nfev_polish,
                          ridge=ridge)
    rms_f = consider(res_f)
    if verbose:
        print(f"[fit] final polish: rms {rms_f:.4e} ({res_f['nfev']} nfev)")

    # 4. duplicate cleanup: near-coincident pairs carry an unpenalized
    # near-cancellation that only shows up between samples / off-window.
    # Merge them, briefly refit, and guarantee the OUTPUT is duplicate-free
    # (the last merge is never followed by a pole refit). A merge is only
    # ever ACCEPTED if it does not degrade the fit; otherwise the pre-merge
    # configuration is kept untouched.
    e_b, w_b, C_b = best["res"]["e"], best["res"]["w"], best["C"]
    rms_b = best["rms"]
    n_merged = 0
    for _ in range(2):
        e_m, w_m, C_m, n_m = merge_duplicates(e_b, w_b, C_b)
        if n_m == 0 or e_m.size < 2:
            break
        res_m = fit_poles(E_fit, B_fit, e_m, w_m, sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut,
                              bounds_fun=bounds_fun, rcond=rcond,
                              max_nfev=nfev_step, **LOOSE)
        C_rm = unpack_coefficients(res_m["X"], chans, p)
        F_rm = basis_apply(
                         lorentzian_basis(E_s, res_m["e"], res_m["w"]),
                         C_rm)
        rms_rm = _quad_rms(F_rm, F_s, qw)
        accept = None
        if rms_rm <= max(2.0 * rms_b, 2.0 * target_rms):
            accept = (res_m["e"], res_m["w"], C_rm, rms_rm)
        else:
            # refit degraded; check the merged coefficients as-is
            F_mo = basis_apply(
                             lorentzian_basis(E_s, e_m, w_m), C_m)
            rms_mo = _quad_rms(F_mo, F_s, qw)
            if rms_mo <= max(2.0 * rms_b, 2.0 * target_rms):
                accept = (e_m, w_m, C_m, rms_mo)
        if accept is None:
            if verbose:
                print(f"[fit] merge degraded rms {rms_b:.4e} -> "
                      f"{min(rms_rm, 1e300):.4e}; keeping pre-merge poles")
            break
        e_b, w_b, C_b, rms_b = accept
        n_merged += n_m

    # 4b. local width floor (non-uniform grids): a pole much narrower
    # than the distance from its center to the nearest sample is
    # invisible to every sample and only produces between-sample spikes.
    # This is the generalization of the tight width floor to grids where
    # the median spacing does not describe the sparse regions. Widen such
    # poles to the local floor, refit, and keep the result only if the
    # sample fit does not degrade (the widened pole was contributing
    # nothing at the samples, so the guard is almost always satisfied).
    n_widened = 0
    if local_floor_frac > 0.0:
        _, hi_1 = bounds_fun(1)
        w_hi = float(hi_1[1])
        for _ in range(2):
            d_near = distance_to_nearest_sample(E_s, e_b)
            floor = np.minimum(local_floor_frac * d_near, w_hi)
            mask = w_b < floor
            if not mask.any():
                break
            w_l = np.maximum(w_b, floor)
            res_l = fit_poles(E_fit, B_fit, e_b, w_l,
                                  sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut,
                                  bounds_fun=bounds_fun, rcond=rcond,
                                  max_nfev=nfev_step, **LOOSE)
            C_l = unpack_coefficients(res_l["X"], chans, p)
            F_l = basis_apply(
                            lorentzian_basis(E_s, res_l["e"], res_l["w"]),
                            C_l)
            rms_l = _quad_rms(F_l, F_s, qw)
            still = subresolution_mask(E_s, res_l["e"], res_l["w"],
                                              frac=local_floor_frac)
            if (rms_l <= max(2.0 * rms_b, 2.0 * target_rms)
                    and not still.any()):
                e_b, w_b, C_b, rms_b = res_l["e"], res_l["w"], C_l, rms_l
                n_widened += int(mask.sum())
                continue
            # refit shrank poles back or degraded the fit: keep the
            # positions, widen once more and only re-solve coefficients
            Phi_c = lorentzian_basis(E_fit, e_b, w_l)
            X_c, _ = solve_coefficients(Phi_c, B_fit,
                                            sample_weights=sw_fit,
                                            rcond=rcond)
            C_c = unpack_coefficients(X_c, chans, p)
            F_c = basis_apply(
                            lorentzian_basis(E_s, e_b, w_l), C_c)
            rms_c = _quad_rms(F_c, F_s, qw)
            if rms_c <= max(2.0 * rms_b, 2.0 * target_rms):
                e_b, w_b, C_b, rms_b = e_b, w_l, C_c, rms_c
                n_widened += int(mask.sum())
            break
        if verbose and n_widened:
            print(f"[fit] widened {n_widened} sub-resolution poles to the "
                  f"local floor (frac={local_floor_frac})")
    history["n_widened_subresolution"] = int(n_widened)
    e_s, w_s, _, n_m = merge_duplicates(e_b, w_b, C_b)
    n_merged += n_m
    if verbose and n_merged:
        print(f"[fit] merged {n_merged} duplicate pairs -> {e_s.size}")

    # final solve at the best poles (weighted; anchors included so the
    # returned model also behaves off-window); guarded: if the merged
    # configuration somehow degrades the fit, revert to the pre-merge one
    def _final_solve(e_in, w_in, ridge_val):
        Phi_ = lorentzian_basis(E_fit, e_in, w_in)
        X_, R_ = solve_coefficients(Phi_, B_fit, sample_weights=sw_fit,
                                        rcond=rcond, ridge=ridge_val)
        C_ = unpack_coefficients(X_, chans, p)
        Phi_r = lorentzian_basis(E_s, e_in, w_in)
        F_ = basis_apply( Phi_r, C_)
        return C_, _quad_rms(F_, F_s, qw)

    C_s, rms = _final_solve(e_s, w_s, ridge)
    # the ridge exists to bound near-null residue excursions; at high pole
    # counts it can also shrink legitimate near-degenerate structure. If
    # it costs more than 2x, redo the final solve without it.
    if rms > 2.0 * rms_b:
        C_r0, rms0 = _final_solve(e_s, w_s, 0.0)
        if rms0 < rms:
            if verbose:
                print(f"[fit] ridge degraded final rms {rms:.4e} -> "
                      f"{rms0:.4e}; solving without ridge")
            C_s, rms = C_r0, rms0
    if n_merged and rms > 4.0 * rms_b:
        if verbose:
            print(f"[fit] merge degraded rms {rms_b:.4e} -> {rms:.4e};"
                  " reverting")
        e_s, w_s = e_b, w_b
        n_merged = 0
        C_s, rms = _final_solve(e_s, w_s, ridge)
    Phi = lorentzian_basis(E_fit, e_s, w_s)
    Phi_real = lorentzian_basis(E_s, e_s, w_s)
    history.update(n_poles=int(e_s.size), rms=float(rms),
                   n_merged_duplicates=int(n_merged))

    psd_info = None
    if psd:
        refined = psd_refine(E_fit, B_fit, e_s, w_s, C_s, chans)
        F_psd = basis_apply( Phi_real, refined["C"])
        refined["rms"] = _quad_rms(F_psd, F_s, qw)

        # PSD-aware pole additions: poles that met the target
        # UNCONSTRAINEDLY rely on inter-pole cancellation, which the
        # projection removes - the projected model can be far off and
        # oscillate around the data in the transport quantities. Add
        # poles at the PROJECTED model's residual peaks until the PSD
        # model meets the target too. Two variants per round: a
        # coefficients-only solve at the augmented poles (poles stay
        # put - no drift back toward the unconstrained optimum) and a
        # short local pole refit; keep whichever projects better.
        n_psd_added = 0
        for _ in range(12):
            if (refined["rms"] <= target_rms
                    or e_s.size >= max_poles):
                break
            new_c, new_w = suggest_pole_insertions(
                E_s, refined["R"][:, :N_real], e_s, w_s,
                max_insert=max(1, min(4, max_poles - e_s.size)))
            if new_c.size == 0:
                break
            centers_a = np.concatenate([e_s, new_c])
            widths_a = np.concatenate([w_s, new_w])
            lo_a, hi_a = bounds_fun(centers_a.size)
            centers_a = np.clip(centers_a, lo_a[0::2] + 1e-9,
                                hi_a[0::2] - 1e-9)
            widths_a = np.clip(widths_a, lo_a[1::2] + 1e-9,
                               hi_a[1::2] - 1e-9)
            cands = []
            # variant A: coefficients only at the augmented poles
            Phi_a = lorentzian_basis(E_fit, centers_a, widths_a)
            X_a, _ = solve_coefficients(Phi_a, B_fit,
                                             sample_weights=sw_fit,
                                             rcond=rcond)
            cands.append(dict(e=centers_a, w=widths_a, X=X_a))
            # variant B: short local pole refit from the same start
            res_b = fit_poles(E_fit, B_fit, centers_a, widths_a,
                                  sample_weights=sw_fit, B_opt=Bc_fit, operator_threshold=operator_threshold, tr_solver=tr_solver, gram_cut=gram_cut,
                                  bounds_fun=bounds_fun, rcond=rcond,
                                  max_nfev=150)
            cands.append(res_b)
            best_a = None
            for res_a in cands:
                C_a = unpack_coefficients(res_a["X"], chans, p)
                refined_a = psd_of(res_a, C_a)
                if best_a is None or refined_a["rms"] < best_a[1]:
                    best_a = (res_a, refined_a["rms"], refined_a, C_a)
            res_a, rms_a, refined_a, C_a = best_a
            if refined_a["rms"] < refined["rms"]:
                e_s, w_s, C_s = res_a["e"], res_a["w"], C_a
                refined = refined_a
                rms, _ = rms_of(res_a)
                Phi_real = lorentzian_basis(E_s, e_s, w_s)
                n_psd_added += int(new_c.size)
                if verbose:
                    print(f"[fit] psd add {new_c.size} -> {e_s.size} "
                          f"pairs, psd rms {refined['rms']:.4e}")
            else:
                break
        history["n_psd_added"] = int(n_psd_added)
        history["n_poles"] = int(e_s.size)
        # keep the PSD model only if it does not explode the error; the
        # unconstrained fit relies on inter-pole cancellation, so forcing
        # per-pole PSD can cost orders of magnitude in accuracy.
        # psd_force accepts the physical (PSD) model unconditionally:
        # PSD residues make the assembled Gamma positive semidefinite
        # EVERYWHERE (between samples and beyond the window), which the
        # transport quantities need - a negative Gamma tail produces
        # unphysical self-energies and spurious transmission peaks.
        refined["applied"] = (psd_force or
                              refined["rms"] <= max(rms * 3.0,
                                                     target_rms * 2.0))
        rms_unc = rms
        if refined["applied"]:
            C_s = refined["C"]
            rms = refined["rms"]
            history["rms_psd"] = rms
        psd_info = refined
        if verbose:
            print(f"[fit] psd: projected rms {refined['rms']:.4e} vs "
                  f"unconstrained {rms_unc:.4e} -> "
                  f"{'APPLIED' if refined['applied'] else 'rejected'}"
                  f"{' (forced)' if psd_force else ''}")

    if verbose:
        w_lo = float(bounds_fun(1)[0][1])
        print(f"[fit] done: {e_s.size} pole pairs, sample rms {rms:.4e} "
              f"(scaled floor w_min={w_lo:.3g}), "
              f"{history.get('n_widened_subresolution', 0)} widened, "
              f"{n_merged} merged"
              + (f", {history.get('n_psd_added', 0)} psd-added"
                 if psd else ""))

    return FitResult(
        e=fd.unscale_centers(e_s),
        w=fd.unscale_widths(w_s),
        C=fd.unscale_coeffs(C_s),
        p=p,
        rms=float(rms),
        data=fd,
        e_scaled=e_s,
        w_scaled=w_s,
        C_scaled=C_s,
        n_poles=int(e_s.size),
        chans=chans,
        history=history,
        psd=psd_info,
    )


def _poles_init_dispatch(init, E_s, F_s, n_init, e_margin=0.0, psd=False,
                         verbose=False):
    # with psd on, prefer the positivity-native NNLS dictionary init:
    # its sparse non-negative solution gives poles that a PSD model
    # can use without inter-pole cancellation (review 4.7)
    if psd:
        try:
            centers, widths = psd_dict_init(E_s, F_s, n_init,
                                           verbose=verbose)
            if centers.size >= 2:
                return centers, widths
        except Exception as exc:
            print(f"[poles_init] psd_dict_init failed ({exc!r}); "
                  "falling back", file=sys.stderr)
    centers, widths = init_poles(E_s, F_s, n_init, method=init,
                                 e_margin=e_margin)
    if centers.size < 2:
        centers, widths = linspace_init(E_s, n_init)
    return centers, widths


# ==========================================================================
# SECTION: TD glue
# ==========================================================================
"""Adaptive level-width sampling refinement — Zandpack wiring.

Pairs `lorentz_fit.sampling` (pure-numpy refinement proposals) with the
TD_Transport / siesta_python machinery that actually evaluates the
level-width function:

    fit -> propose insertions -> rebuild contour -> rerun TBtrans
    -> read -> refit -> ... -> (optional) propose removals -> final fit

Designed to slot into a 3.py-style workflow:

    R = load_pickle('tmp.gz')
    D = pickle.load(open('Device.SiP','rb'))
    D.mpi = 'mpirun '
    out = refine_loop(R, D, read_kwargs=dict(sub_orbital=orbs,
                                             less_memory=False))

Import order: this module never puts the repo root on sys.path
(lorentz_fit is loaded by explicit path), so importing it can never
shadow the installed Zandpack/siesta_python packages. Example scripts
that import it should add the repo root temporarily and remove it right
after the import:

    sys.path.insert(0, ROOT)
    import refine_sampling
    sys.path.remove(ROOT)
"""

import gzip
import lzma
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# data access (mirrors bench/extractpy, which must stay import-free
# of lorentz_fit for its own load-order reasons)
# --------------------------------------------------------------------------

def load_pickle(path):
    """gzip (.gz) or lzma (.xz) pickle loader."""
    opener = gzip.open if str(path).endswith(".gz") else lzma.open
    with opener(path, "rb") as f:
        return pickle.load(f)


def block_sizes(G):
    sizes = {}
    for bi, (I, J) in enumerate(G.inds):
        blk = G.vals[bi][0]
        if int(I) == int(J):
            sizes[int(I)] = blk.shape[-1]
    for bi, (I, J) in enumerate(G.inds):
        blk = G.vals[bi][0]
        sizes.setdefault(int(I), blk.shape[-2])
        sizes.setdefault(int(J), blk.shape[-1])
    return sizes


def extract_dense(R, lead, ik):
    """Dense (E, F) for one lead and k-point; E (N,), F (N, p, p).

    The stored Gammas may keep only the LOWER TRIANGLE of the block
    structure (e.g. VerySmallRSSE: blocks (0,0), (1,0), (1,1) with no
    (0,1) partner). A naive Hermitization then HALVES the off-diagonal
    blocks - the fit would model Gamma_10 / 2 and every downstream
    transport quantity comes out wrong. Missing partner blocks are
    therefore completed with the conjugate transpose BEFORE the final
    symmetrization (which is then a numerical no-op).
    """
    sidx = R.sampling_idx[lead]
    E = np.asarray(R.Contour[sidx]).real
    G = R.Nonortho_Gammas[lead]
    sizes = block_sizes(G)
    order = sorted(sizes)
    offsets = {}
    p = 0
    for I in order:
        offsets[I] = p
        p += sizes[I]

    F = np.zeros((len(sidx), p, p), dtype=complex)
    have = {(int(I), int(J)) for I, J in G.inds}
    for bi, (I, J) in enumerate(G.inds):
        blk = G.vals[bi][ik][sidx]
        si, sj = offsets[int(I)], offsets[int(J)]
        F[:, si:si + blk.shape[-2], sj:sj + blk.shape[-1]] = blk
    # complete missing transpose partners
    for (I, J) in have:
        if (J, I) not in have and I != J:
            bi = [k for k, (a, b) in enumerate(G.inds)
                  if int(a) == I and int(b) == J][0]
            blk = G.vals[bi][ik][sidx]
            si, sj = offsets[int(I)], offsets[int(J)]
            F[:, sj:sj + blk.shape[-1], si:si + blk.shape[-2]] = \
                np.conj(np.swapaxes(blk, -1, -2))
    F = 0.5 * (F + np.conj(np.swapaxes(F, -1, -2)))
    return E, F


def current_lines(R):
    """Per-lead current sampling energies (real parts of the lead lines)."""
    return [np.asarray(R.Contour[R.sampling_idx[a]]).real.copy()
            for a in range(R.num_leads)]


# --------------------------------------------------------------------------
# evaluation: rebuild contour, rerun TBtrans, read
# --------------------------------------------------------------------------

_TS_TBT_SNAPSHOT = None


def resample(R, D, lines, eta=1e-2, N_F=2, tol=1e-6, mpi=None,
             read_kwargs=None, analyze=False, runner=None):
    """Rebuild the contour from per-lead real energy arrays, re-evaluate
    the level-width functions in Dev.dir and read them back into R.

    lines : list of per-lead 1D real arrays (shared grid allowed)
    eta   : imaginary offset of the sampling line (as in the 2.py scripts)
    runner : optional callable(R, D) replacing the default evaluation
        step (contour -> fdf -> TBtrans). Used for setups where the
        self-energies are computed differently, e.g. the spin-polarized
        D.mimic_tbtrans_spin(R.Contour) path.
    """
    global _TS_TBT_SNAPSHOT
    if mpi is not None:
        D.mpi = mpi
    stacked = np.vstack([np.asarray(ln, dtype=complex) + 1j * eta
                         for ln in lines])
    R.Make_Contour(stacked, N_F, tol=tol)
    if runner is not None:
        runner(R, D)
    else:
        # write_more_fdf APPENDS, so repeated rounds would stack duplicate
        # TBT flag blocks in TS_TBT.fdf; snapshot the pristine file
        # (written once by the original 2.py analyze step) and restore it
        # per round
        ts_path = os.path.join(getattr(D, "dir", "."), "TS_TBT.fdf")
        if _TS_TBT_SNAPSHOT is None and os.path.exists(ts_path):
            with open(ts_path) as f:
                _TS_TBT_SNAPSHOT = f.read()
        D.custom_tbtrans_contour = R.Contour
        D.fdf()
        if _TS_TBT_SNAPSHOT is not None:
            with open(ts_path, "w") as f:
                f.write(_TS_TBT_SNAPSHOT)
        if analyze:
            D.run_tbt_analyze_in_dir()
        D.run_tbtrans_in_dir(DOS_GF=True, overwrite_prev=True)
    R.Device = D
    R.read_data(**(read_kwargs or {}))


# --------------------------------------------------------------------------
# the refinement loop
# --------------------------------------------------------------------------

def inject_fits(R, fits, lowdin_convert=True, equal_nl="all"):
    """Inject lorentz_fit results into a TD_Transport object so the
    Zandpack post-processing (find_correction, Renormalise_H,
    Inspect_transmission_from_hilbert_transform,
    Inspect_dos_from_hilbert_transform, tofile) works exactly as after
    the old R.Fit pipeline.

    fits : {(lead, ik): FitResult}. Per (lead, ik) the fit has its own
    poles; the Zandpack container supports per-ik pole arrays, so the
    per-ik pole sets are written directly (pads the smaller sets with
    zero-residue rows when they differ).

    equal_nl : the REST OF ZANDPACK requires a single common pole
    count — R.NumL is ONE scalar and the container helpers
    (add_broadening, add_move_center, setLval's default iL) index
    every lead with it. Default 'all' therefore pads every (lead, ik)
    pole array to the GLOBAL maximum with inert zero-residue
    duplicates of pole 0 (identical assembled Gamma — proven on the
    spin systems). 'lead' reproduces the old per-lead widths and is
    kept ONLY for legacy bit-comparisons; it breaks the R.NumL
    contract when the leads' maxima differ. See also
    equalize_pole_counts (post-fit variant, e.g. before pickling
    the fits themselves).

    The COEFFICIENTS are written through the class's own API:
    R.setLval(lead, I, J, i, j, val, ik, iL) per (block, sub-block)
    component - setLval fills the Hermitian partner block (J,I) with
    the conjugate automatically - followed by R.FitNO2O() to populate
    the orthogonalized fitted_lorentzians. The dense p x p fit
    matrices only have to be mapped back through the I,J block and
    i,j sub-block indices (the same layout extract_dense packed them
    with). The container skeleton (zeros + ei/gamma + the Hermitian
    partner blocks that Block() needs to exist) is created first,
    since Block() returns None for absent blocks and setLval could
    not write through it.

    The stored Gammas may keep only the LOWER TRIANGLE of the block
    structure (e.g. VerySmallRSSE: no (0,1) partner). The old R.Fit
    containers carry the FULL block set - the transport machinery
    (Blocksparse2Numpy -> dense Gamma) needs the Hermitian completion -
    so the injected containers get the missing partner blocks too,
    filled by setLval with the conjugate-transposed residues.
    """
    created = []
    # R.Fit normally creates these containers; a fresh read_data-only
    # TDT object does not have them yet
    if not hasattr(R, "NO_fitted_lorentzians"):
        R.NO_fitted_lorentzians = []
    if not hasattr(R, "fitted_lorentzians"):
        R.fitted_lorentzians = []
    while len(R.NO_fitted_lorentzians) < R.num_leads:
        R.NO_fitted_lorentzians.append(None)
    while len(R.fitted_lorentzians) < R.num_leads:
        R.fitted_lorentzians.append(None)

    NL_all = 0
    NL_per_lead = {}
    for lead in range(R.num_leads):
        n_ik_l = len(R.Nonortho_Gammas[lead].vals[0])
        NL_per_lead[lead] = max(fits[(lead, ik)].n_poles
                                for ik in range(n_ik_l))
        NL_all = max(NL_all, NL_per_lead[lead])
    for lead in range(R.num_leads):
        G_samp = R.Nonortho_Gammas[lead]
        n_ik = G_samp.vals[0].shape[0]
        # Zandpack REQUIRES a single common pole count: the container
        # helpers (add_broadening, add_move_center, setLval's default
        # iL) all index with the one scalar R.NumL, so every k-point
        # of every lead must carry exactly NL_all poles. Default
        # equal_nl='all' pads every container to the GLOBAL max with
        # inert zero-residue duplicates of pole 0 (proven mechanism —
        # identical assembled Gamma). equal_nl='lead' reproduces the
        # OLD per-lead widths and is kept ONLY for legacy
        # bit-comparisons; it violates the R.NumL contract when the
        # leads' maxima differ.
        if equal_nl == "all":
            NL = NL_all
        elif equal_nl == "lead":
            NL = NL_per_lead[lead]
        else:
            raise ValueError("inject_fits: equal_nl must be 'lead' "
                             "or 'all'")
        # dense-basis offsets, same layout as extract_dense
        sizes, offsets, p = {}, {}, 0
        for bi, (I, J) in enumerate(G_samp.inds):
            blk = G_samp.vals[bi][0]
            if int(I) == int(J):
                sizes[int(I)] = blk.shape[-1]
        for bi, (I, J) in enumerate(G_samp.inds):
            blk = G_samp.vals[bi][0]
            sizes.setdefault(int(I), blk.shape[-2])
            sizes.setdefault(int(J), blk.shape[-1])
        for I in sorted(sizes):
            offsets[I] = p
            p += sizes[I]

        Ei = np.zeros((n_ik, NL))
        Gi = np.zeros((n_ik, NL))
        for ik in range(n_ik):
            res = fits[(lead, ik)]
            Ei[ik, :res.n_poles] = res.e
            Gi[ik, :res.n_poles] = res.w
            # pad rows: duplicate pole 0 (its coefficients are zeroed
            # below, so the duplicate contributes nothing)
            Ei[ik, res.n_poles:] = res.e[0]
            Gi[ik, res.n_poles:] = res.w[0]

        # full (Hermitian-completed) block set. The values go in
        # through R.setLval below; the skeleton is initialized with
        # ONES (as Zandpack's initialize_lorentzian_fit does), NOT
        # zeros: block_sparse.info() marks all-val-zero blocks as
        # is_zero=0 at construction, and a zero-filled container
        # would be skipped by later block_MATMUL products (BDot),
        # e.g. FitNO2O's Lowdin transform.
        have = [(int(I), int(J)) for I, J in G_samp.inds]
        pairs = list(have)
        for (I, J) in have:
            if (J, I) not in pairs:
                pairs.append((J, I))

        inds, vals = [], []
        for (I, J) in pairs:
            ni, nj = sizes[I], sizes[J]
            blk = np.ones((n_ik, NL, ni, nj), dtype=np.complex128)
            inds.append(np.array([I, J], dtype=np.int32))
            vals.append(blk)

        G_fit = type(G_samp)(inds, vals, G_samp.Block_shape)
        G_fit.Lorentzian_basis = True
        G_fit.ei = Ei
        G_fit.gamma = Gi

        R.NO_fitted_lorentzians[lead] = G_fit
        created.append(G_fit)

        # write the coefficients through the class API: one setLval
        # call per (block, sub-block) component; setLval fills the
        # Hermitian partner (J,I) with the conjugate itself, so only
        # one orientation of each block pair is written explicitly.
        # iL is passed explicitly (poles may differ per lead; setLval's
        # default iL uses the single scalar R.NumL).
        iL = np.arange(NL)
        prim = {}
        for (I, J) in pairs:
            prim.setdefault((min(I, J), max(I, J)), (I, J))
        for ik in range(n_ik):
            res = fits[(lead, ik)]
            vec = np.zeros(NL, dtype=np.complex128)
            for (I, J) in prim.values():
                ni, nj = sizes[I], sizes[J]
                si, sj = offsets[I], offsets[J]
                for i in range(ni):
                    for j in range(nj):
                        vec[:] = 0.0
                        vec[:res.n_poles] = res.C[:, si + i, sj + j]
                        R.setLval(lead, I, J, i, j, vec,
                                  ik=ik, iL=iL)

    # setLval's default iL and the add_broadening/add_move_center
    # helpers index with R.NumL; use the largest pole count across
    # leads (per-lead sets are written with explicit iL above).
    R.NumL = int(NL_all)

    if lowdin_convert:
        R.FitNO2O()   # populate fitted_lorentzians (orthogonal basis)
    return created


def sampled_transmission_dos(R):
    """Transmission and DOS from the SAMPLED self-energies - the exact
    reference for fit comparisons.

    The stored R.self_energies live in the SAME sub-orbital space as
    the fitted containers (read_data's sub_orbital selection), so this
    is the apples-to-apples reference: no fit, no correction, same
    orbital truncation. (TBtrans's own transmission includes ALL
    orbitals, so it is NOT comparable whenever sub_orbital removed
    transmission-carrying orbitals - e.g. zgnr below E = -3.)

    Returns (E_real, T, dos): T and dos evaluated at the sampled
    contour energies (which carry the contour's imaginary offset),
    k-averaged with the stored weights.
    """
    S = R.bs2np(R.BTD_overlap)
    H = R.bs2np(R.Nonortho_Hamiltonian)
    sidx = R.sampling_idx[0]
    z = np.asarray(R.Contour[sidx])
    SEs = [np.asarray(R.bs2np(se))[:, sidx] for se in R.self_energies]
    Eg = z[np.newaxis, :, np.newaxis, np.newaxis]
    G = S * Eg - H - sum(SEs)
    G = np.linalg.inv(G)
    Gd = np.conj(np.swapaxes(G, -1, -2))
    Gams = [1j * (SEs[a] - np.conj(np.swapaxes(SEs[a], -1, -2)))
            for a in range(len(SEs))]
    T = np.trace(Gams[0] @ G @ Gams[-1] @ Gd, axis1=2, axis2=3)
    dos = -np.trace(G, axis1=2, axis2=3).imag / np.pi
    wk = np.asarray(R.tbtwkpt)
    T = (T * wk[:, None]).sum(axis=0).real
    dos = (dos * wk[:, None]).sum(axis=0)
    return np.asarray(z).real, T, dos


class CombinedFit:
    """A sum of two shared-pole Lorentzian families (base fit + a
    small correction fit of the residual), exposed as one model. The
    concatenated pole set is still shared across all matrix entries,
    so inject_fits can build one Zandpack container from it."""

    def __init__(self, parts):
        self.e = np.concatenate([p.e for p in parts])
        self.w = np.concatenate([p.w for p in parts])
        self.C = np.concatenate([p.C for p in parts])
        self.n_poles = int(self.e.size)
        self.rms = None
        self.psd = None
        self.history = {"boost_parts": [p.n_poles for p in parts]}

    def evaluate(self, E):
        Phi = lorentzian_basis(np.asarray(E, dtype=float),
                                           self.e, self.w)
        return basis_apply(Phi, self.C)


class PaddedFit:
    """A fit viewed with extra INERT poles so that every (lead, ik)
    can carry the same pole count (multi-k-point systems). Padded
    poles carry ZERO residues; their (e, w) is either a duplicate of
    the fit's own pole 0 ('dup' mode: real width, NaN-safe by
    construction — the mechanism inject_fits has always used for
    per-ik padding) or a zero-WIDTH pole placed far outside the
    energy window ('zero' mode: the Lorentzian w^2/((E-e)^2 + w^2)
    is identically 0 there; the center sits at max|e| + 1e3 so no
    physical evaluation grid can hit the 0/0 point at E == e).
    evaluate() delegates to the base fit, so the padded model is
    EXACTLY the input model; all other attributes delegate too."""

    def __init__(self, base, e, w, C, mode):
        self._base = base
        self._pad_mode = mode
        self.e = e
        self.w = w
        self.C = C
        self.n_poles = int(e.size)

    def evaluate(self, E):
        # padded poles are inert -> identical to the base model
        return self._base.evaluate(E)

    def __getattr__(self, name):
        # guard: any attribute access before _base is set (e.g. the
        # __setstate__ probe while UNPICKLING) must raise, not recurse
        if "_base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._base, name)


def pad_poles(fit, n_target, mode="dup"):
    """Return `fit` padded to exactly `n_target` poles with inert
    zero-residue poles (see PaddedFit). `fit` itself is untouched;
    fewer poles than n_target returns `fit` unchanged.

    mode='dup'  : duplicate the fit's pole 0 (real width, zero
                  residue) — NaN-safe everywhere, the proven
                  inject_fits mechanism.
    mode='zero' : append poles with w=0 and e = max|e| + 1e3 (the
                  basis is identically 0; far placement avoids the
                  0/0 at E == e). Do not evaluate a grid point
                  exactly at that center.
    """
    n = int(fit.n_poles)
    n_target = int(n_target)
    if n_target <= n:
        return fit
    k = n_target - n
    if mode == "dup":
        e_pad = np.full(k, float(fit.e[0]))
        w_pad = np.full(k, float(fit.w[0]))
    elif mode == "zero":
        e_pad = np.full(k, float(np.max(np.abs(fit.e))) + 1e3)
        w_pad = np.zeros(k)
    else:
        raise ValueError(f"pad_poles: unknown mode {mode!r} "
                         "(use 'dup' or 'zero')")
    C_pad = np.zeros((k,) + np.asarray(fit.C).shape[1:],
                     dtype=np.asarray(fit.C).dtype)
    return PaddedFit(fit,
                     np.concatenate([np.asarray(fit.e), e_pad]),
                     np.concatenate([np.asarray(fit.w), w_pad]),
                     np.concatenate([np.asarray(fit.C), C_pad], axis=0),
                     mode)


def equalize_pole_counts(fits, mode="dup", n_target=None):
    """Equalize pole counts across ALL (lead, ik) fits of a multi-k
    system — post-fit processing, purely additive.

    Every fit is padded to a common count (default: the maximum in
    `fits`; or `n_target`) with inert zero-residue poles, so the
    injected Zandpack containers carry the SAME pole-set size for
    every k-point AND every lead. The padded models evaluate
    identically to the inputs. Returns a NEW dict; pass it to
    inject_fits as usual (its per-ik padding then finds equal counts
    and adds nothing).

    fits : {(lead, ik): FitResult or CombinedFit}
    mode : 'dup' (default, NaN-safe duplicate of pole 0) or 'zero'
           (zero-width poles far outside the window) — see pad_poles.
    """
    if n_target is None:
        n_target = max(int(f.n_poles) for f in fits.values())
    return {key: pad_poles(f, n_target, mode=mode)
            for key, f in fits.items()}


def fit_boosted(E, F, base_kw, n_res=8, res_kw=None):
    """Two-path fit (user recipe): a first (adaptive, PSD) fit, then a
    SECOND fit of the residual (samples - first fit) with a much
    SMALLER pole budget, added back to the first. The residual is
    smooth and low-rank once the bulk is captured, so a small fresh
    pole optimization from its own basin corrects sample-level errors
    (and spurious structure) that the first pass could not. The
    correction fit is PSD-forced too, so the SUM stays PSD everywhere.

    base_kw : fit kwargs for the first pass (adaptive; psd=True,
        psd_force=True recommended).
    n_res   : correction pole pairs (significantly fewer than the
        first pass - 8 is a good default).
    res_kw  : optional overrides for the correction fit.
    Returns a CombinedFit (e/w/C concatenated, injectable).
    """
    res_kw = dict(base_kw or {})
    res_kw.update(res_kw or {})
    res_kw["n_poles"] = int(n_res)
    # target_rms=0.0 + no max_poles: the correction fit stays at
    # exactly n_res poles (no adaptive insertion growth).
    res_kw["target_rms"] = 0.0
    res_kw.pop("max_poles", None)
    res_kw.pop("init_poles", None)
    res_kw["psd"] = True
    res_kw["psd_force"] = True

    E = np.asarray(E, dtype=float)
    F = np.asarray(F)
    r1 = fit(E, F, **base_kw)
    resid = F - r1.evaluate(E)
    r2 = fit(E, resid, **res_kw)
    comb = CombinedFit([r1, r2])
    comb.rms = float(frobenius_rms_error(
        comb.evaluate(E), F))
    return comb


def _fit_one(E, F, fit_kw):
    p = F.shape[1]
    # psd=True + psd_force=True is the DEFAULT of the refinement
    # workflow: PSD residue matrices make the assembled Gamma PSD
    # everywhere - between samples and beyond the window, by
    # construction. This is required for the transport quantities: an
    # unconstrained (Hermitian-only) fit can have a Gamma that goes
    # NEGATIVE outside the sampling window (observed on VerySmallRSSE:
    # Tr(Gamma) = -1.3 at +7), which produces unphysical self-energies
    # and spurious transmission resonances near the window edge.
    # psd_force accepts the rms cost of the physical model; pass
    # fit_kw=dict(psd_force=False) to restore the accuracy guard.
    kw = dict(n_init=24, max_poles=100, target_rms=2e-3, psd=True,
              psd_force=True,
              channel_tol=1e-4 if p > 32 else 0.0,
              # width policy: no very broad poles (a window-span pole is
              # an unconstrained background with a KK ramp no constant
              # correction absorbs), and a floor of 3x the data's eta so
              # resolution-limit poles cannot make the fit oscillate
              # around the sampling function
              w_max_raw=0.5, eta_floor_mult=5.0)
    kw.update(fit_kw)
    return fit(E, F, **kw)


def refine_loop(R, D, read_kwargs=None, n_rounds=3, insert_tol=5e-3,
                max_add_per_round=48, remove_tol=1e-2, do_remove=True,
                fit_kw=None, initial_lines=None, runner=None,
                eta=1e-2, verbose=True,
                derivative_only=False, deriv_tol=3.0):
    """Run the adaptive refinement loop on a loaded TD_Transport object.

    initial_lines : optional per-lead real energy arrays to start from
        (e.g. a coarsened version of the current sampling). The loop then
        refines upward from there; None keeps the stored sampling.

    derivative_only : FIT-FREE mode. No fits run during the insertion
        rounds; the insertion proposals come purely from the derivative
        of the abs-summed level-width function (d(sum_ij|F_ij|)/dE -
        see lorentz_fit.sampling.suggest_insertions_derivative). Steep
        flanks get density, at a tiny fraction of the cost. The
        leave-one-out removal stage is fit-free in both modes, and the
        FINAL fits (the returned dict) still run once the grid settles.

    Per round: fit every (lead, k-point), propose insertions from the
    fit + interpolation discrepancy, merge the proposals of all leads
    into the shared contour (TBtrans evaluates both leads at every
    point anyway), rerun TBtrans and read. After the insertion rounds,
    one optional removal round drops samples that the leave-one-out test
    flags as redundant for EVERY lead and k-point.

    Returns a dict:
      fits      : {(lead, ik): FitResult} at the final grid
      history   : per-round record (grid sizes, rms per lead/ik)
      removed   : indices dropped from the final line
    """
    fit_kw = fit_kw or {}
    # the data was sampled on E + i*eta: the width floor forbids poles
    # narrower than that broadening (review 4.3)
    fit_kw.setdefault("eta_raw", eta)
    log = print if verbose else (lambda *a, **k: None)

    log(f"[refine] refine_loop: {R.num_leads} leads, "
        f"eta={eta:g}, n_rounds={n_rounds}, insert_tol={insert_tol:g}, "
        f"do_remove={do_remove}"
        + (f", remove_tol={remove_tol:g}" if do_remove else ""))
    log(f"[refine] fit kwargs: "
        f"{ {k: v for k, v in fit_kw.items()} }")

    if initial_lines is not None:
        start = np.unique(np.concatenate(
            [np.asarray(ln, dtype=float) for ln in initial_lines]))
        log(f"[refine] starting from coarse grid with {start.size} points")
        resample(R, D, [start] * R.num_leads, read_kwargs=read_kwargs,
                 runner=runner, eta=eta)
        lines = [start.copy() for _ in range(R.num_leads)]
    else:
        lines = current_lines(R)
    # k-points of the stored Gammas: vals[block][ik][sample]
    G0 = R.Nonortho_Gammas[0]
    n_ik = len(G0.vals[0])

    if derivative_only:
        log("[refine] DERIVATIVE-ONLY mode: no fits during the "
            "insertion rounds (proposals from d(sum_ij |F_ij|)/dE, "
            f"deriv_tol={deriv_tol:g})")

    history = []
    for rnd in range(n_rounds):
        proposals = [[] for _ in lines]
        rms_report = []
        if derivative_only:
            log(f"[refine] === round {rnd}: derivative proposals on "
                f"{lines[0].size} points (no fits) ===")
            for a in range(R.num_leads):
                for ik in range(n_ik):
                    E, F = extract_dense(R, a, ik)
                    share = max(4, max_add_per_round
                                // (R.num_leads * max(n_ik, 1)))
                    ins = suggest_insertions_derivative(
                        E, F, tol=deriv_tol, max_add=share)
                    proposals[a] = proposals[a] + list(ins)
                    log(f"[refine] round {rnd} lead{a} ik{ik}: "
                        f"{len(ins)} derivative insertion(s)"
                        + (f" at {np.round(np.sort(ins), 3).tolist()}"
                           if len(ins) else ""))
        else:
            log(f"[refine] === round {rnd}: fitting on {lines[0].size} "
                f"points ===")
            for a in range(R.num_leads):
                for ik in range(n_ik):
                    E, F = extract_dense(R, a, ik)
                    res = _fit_one(E, F, fit_kw)
                    rms_report.append((a, ik, res.rms))
                    share = max(4, max_add_per_round
                                // (R.num_leads * max(n_ik, 1)))
                    ins = suggest_insertions(
                        E, F, tol=insert_tol, max_add=share,
                        poles=(res.e, res.w))
                    proposals[a] = proposals[a] + list(ins)
                    log(f"[refine] round {rnd} lead{a} ik{ik}: "
                        f"n={res.n_poles} rms={res.rms:.4g} -> "
                        f"{len(ins)} insertion(s)"
                        + (f" at {np.round(np.sort(ins), 3).tolist()}"
                           if len(ins) else ""))
        proposals = [np.unique(np.asarray(pr, dtype=float))
                     for pr in proposals]

        # shared grid: TBtrans evaluates all leads at every point, so the
        # union of the per-lead proposals is the new sampling line
        union = np.unique(np.concatenate([lines[a] for a in range(len(lines))]
                                         + proposals))
        if union.size == lines[0].size and np.allclose(union, lines[0]):
            log(f"[refine] round {rnd}: no insertions proposed; stopping")
            break
        n_new = int(union.size - lines[0].size)
        log(f"[refine] round {rnd}: inserting {n_new} point(s) "
            f"(per-lead proposals: "
            f"{[len(pr) for pr in proposals]}); grid "
            f"{lines[0].size} -> {union.size}; rerunning evaluation")
        resample(R, D, [union] * len(lines), read_kwargs=read_kwargs,
                 runner=runner, eta=eta)
        lines = [union.copy() for _ in lines]
        history.append(dict(round=rnd, n_points=int(union.size),
                            derivative_only=bool(derivative_only),
                            rms={f"lead{a}_ik{ik}": float(r)
                                 for a, ik, r in rms_report}))

    # removal round: drop points redundant for every lead and k-point
    removed = np.array([], dtype=int)
    if do_remove:
        line = lines[0]
        rem_sets = []
        for a in range(R.num_leads):
            for ik in range(n_ik):
                E, F = extract_dense(R, a, ik)
                rem = suggest_removals(E, F, tol=remove_tol)
                rem_sets.append(set(rem.tolist()))
                log(f"[refine] removal candidates lead{a} ik{ik}: "
                    f"{len(rem)} of {len(line)}")
        common = set(range(len(line)))
        for s in rem_sets:
            common &= s
        removed = np.array(sorted(common), dtype=int)
        if removed.size:
            keep = np.setdiff1d(np.arange(len(line)), removed)
            new_line = line[keep]
            log(f"[refine] removal: {removed.size} of {len(line)} points "
                f"redundant for all leads/k; rerunning TBtrans with "
                f"{new_line.size} points")
            resample(R, D, [new_line] * len(lines), read_kwargs=read_kwargs,
                     runner=runner, eta=eta)
            lines = [new_line.copy() for _ in lines]
        else:
            log("[refine] removal: no common redundant points")

    fits = {}
    for a in range(R.num_leads):
        for ik in range(n_ik):
            E, F = extract_dense(R, a, ik)
            fits[(a, ik)] = _fit_one(E, F, fit_kw)
            log(f"[refine] final fit lead{a} ik{ik}: "
                f"n={fits[(a, ik)].n_poles} "
                f"rms={fits[(a, ik)].rms:.4g}")

    return dict(fits=fits, history=history, removed=removed,
                lines=lines)
