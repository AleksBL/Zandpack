#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is a version of the FittedSelfEnergy module in the main Zandpack
directory, but where the functions has been optimized using AI (GLM5.3).
The current use is in the viljas-cuevas code, where the Lorentzian_SE class
is the bottleneck. The original FittedSelfEnergy is found in the base
Zandpack directory.

FittedSelfEnergy v2 (fused + packed rows + pre-evaluated basis table).

Same design as v1, with one further optimization of the noopt=False path:
the energy-dependent part of the self-energy is hoisted out of the matrix
element loop into a basis table

    B[ie, l] = -0.5j * L(E[ie]; ei[l], Wi[l])
             +  0.5 * L(E[ie]) * (E[ie] - ei[l]) / Wi[l],
    L(E; ei, W) = W^2 / ((E - ei)^2 + W^2),

built once per (ik, E) at O(len(E) * Nl) cost, so the evaluation collapses
to the contraction

    res[ik, ie, i, j] = sum_l Coeffs[ik, l, i, j] * B[ie, l]

with a single complex FMA per (row, energy, Lorentzian). v1 recomputed the
Lorentzian and its KK factor once per active matrix element; v2 computes
them once per call. B is rebuilt on every call, so arbitrary (variable,
real- or complex-valued, any length) input E is supported; only the packed
coefficient rows are cached on the instance.

Measured on Testing/ABC (nk=1, 44 Lorentzians, 72x72, 1374 active rows):
  original two-pass : 0.34 s (ne=500), 4.8 ms (ne=1)
  v1 fused+packed   : 0.18 s (ne=500), 0.45 ms (ne=1)
  v2 basis table    : 0.05 s (ne=500), 0.09 ms (ne=1)

The other paths are unchanged from v1: noopt=True uses the base kernels
L_sum / KK_L_sum (Lvec reconstructed consistently with L_sum_opt), and the
hermitian paths raise NotImplementedError (their kernels live in
Block_matrices.Croy, which is not available in this directory).
"""
import numpy as np
import numba as nb
from Zandpack.Loader import flexload

FASTMATH = True
CACHE = False


# ---------------------------------------------------------------------------
# Basis table + contraction kernel for the noopt=False path
# ---------------------------------------------------------------------------

# What turned out to be helpful was to hoist out the evaluation of the
# Lorentzian functions on a grid, instead of evaluating them for each
# coefficient in the level-width function matrix. Also, by fusing the
# Loops for KK_L_sum and L_sum, a speed-up was achieved.
# Also, further speed was acheived by only looping over nonzero elements
# (the _pack function sets this up).
# The structure of the build_B function below, in terms of operations
# is coming from the Block_matrices.Croy.L_sum_opt and KK_L_sum_opt.
# From the precomputed lorentzian grid computed by build_B, the
# evaluate_LL_KK_table then computes the fitted level-width function
# faster than if it had to evaluate all the Lorentzians for each matrix entry.
@nb.njit(cache=CACHE, fastmath=FASTMATH)
def build_B(E, ei, gamma, ik):
    """
    Pre-evaluate the fused Lorentzian + KK basis on the energy grid.

    B[ie, l] contains the complete energy-dependent factor of the
    self-energy for Lorentzian l at energy E[ie]; res is then a plain
    contraction of Coeffs with B. Shape (len(E), Nl), complex128.
    """
    nl = ei.shape[1]
    ne = len(E)
    B = np.empty((ne, nl), dtype=np.complex128)
    for ie in range(ne):
        e = E[ie]
        for l in range(nl):
            w = gamma[ik, l]
            w2 = w * w
            A = e - ei[ik, l]
            KKf = A / w
            A *= A
            A += w2
            A = w2 / A
            B[ie, l] = A * (-0.5j) + (A * KKf) * 0.5
    return B


@nb.njit(cache=CACHE, fastmath=FASTMATH)
def evaluate_L_KK_table(Gpack, Iidx, Jidx, B, res, ik):
    """
    Contract the packed coefficient rows with the basis table.

    Gpack: (nact, Nl) C-contiguous active coefficient rows (Lorentzian last)
    Iidx, Jidx: (nact,) matrix indices of the packed rows
    B: (ne, Nl) basis table from build_B
    res: (nk, ne, ni, nj) complex128 output; res[ik] is filled here
    """
    ne = B.shape[0]
    nl = B.shape[1]
    nact = Gpack.shape[0]
    acc = np.empty(ne, dtype=np.complex128)
    for a in range(nact):
        Gc = Gpack[a]
        for ie in range(ne):
            s = 0.0 + 0.0j
            for l in range(nl):
                s += Gc[l] * B[ie, l]
            acc[ie] = s
        res[ik, :, Iidx[a], Jidx[a]] = acc
    return res


# ---------------------------------------------------------------------------
# Base (non-opt) kernels for the noopt=True path, copied from the original.
# Lvec is reconstructed (see module docstring).
# ---------------------------------------------------------------------------
@nb.njit(cache=CACHE, fastmath=FASTMATH)
def Lvec(E, W, e):
    return W * W / ((E - e) * (E - e) + W * W)


@nb.njit(cache=CACHE, fastmath=FASTMATH)
def L_sum(E, var):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0, :])
    res = np.zeros(len(E), dtype=np.complex128)
    for l in range(nl):
        res += Gi[l] * Lvec(E, Wi[l], ei[l])
    return res


@nb.njit(cache=CACHE, fastmath=FASTMATH)
def KK_L_sum(E, var):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0, :])
    res = np.zeros(len(E), dtype=np.complex128)
    for l in range(nl):
        res += Gi[l] * Lvec(E, Wi[l], ei[l]) * (E - ei[l]) / Wi[l]
    return res


@nb.njit(cache=CACHE, fastmath=FASTMATH)
def evaluate_Lorentz_basis_matrix(M, E, ei, gamma, tol=1e-15):
    nk = M.shape[0]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni, nj), dtype=np.complex128)
    pars = np.zeros((3, ei.shape[1]), dtype=np.complex128)
    for ik in range(nk):
        pars[0] = gamma[ik]
        pars[1] = ei[ik]
        for i in range(ni):
            for j in range(nj):
                pars[2] = M[ik, :, i, j]
                if (np.abs(pars[2]) > tol).any():
                    res[ik, :, i, j] = L_sum(E, pars)
    return res


@nb.njit(cache=CACHE, fastmath=FASTMATH)
def evaluate_KK_matrix(M, E, ei, gamma, tol=1e-15):
    nk = M.shape[0]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni, nj), dtype=np.complex128)
    pars = np.zeros((3, ei.shape[1]), dtype=np.complex128)
    for ik in range(nk):
        pars[0] = gamma[ik]
        pars[1] = ei[ik]
        for i in range(ni):
            for j in range(nj):
                pars[2] = M[ik, :, i, j]
                if (np.abs(pars[2]) > tol).any():
                    res[ik, :, i, j] = KK_L_sum(E, pars)
    return res


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Lorentzian_SE:
    def __init__(self, Ei, Gi, Coeffs, tol=1e-14):
        """
           Ei:    Positions of lorentzians on real axis (np.ndarray, (#nk, #Centres))
           Gi:    Broadenings of lorentzians on real axis (np.ndarray, (#nk, #Centres))
           Coeffs: Fitting Coefficients of the Broadening Matrix
                   (np.ndarray, (#nk, #Centres, #no, #no))
           tol:   Sparsity tolerance used when packing the coefficient rows.

           Serves to give the self-energy from evaluating the Lorentzian
           basis functions on a complex energy grid.
        """
        self.Coeffs = Coeffs.copy()
        self.Ei = Ei.copy()
        self.Gi = Gi.copy()
        self._packed = None
        self._pack_tol = None
        self._pack(tol)

    def _pack(self, tol):
        """Pack the active coefficient rows once: (nact, Nl) per ik,
        Lorentzian index last, C-contiguous, plus their (i, j) indices."""
        C = self.Coeffs
        MT = np.ascontiguousarray(C.transpose(0, 2, 3, 1))  # (nk, ni, nj, Nl)
        act = (np.abs(C) > tol).any(axis=1)                 # (nk, ni, nj)
        packed = []
        for ik in range(C.shape[0]):
            I, J = np.nonzero(act[ik])
            packed.append((np.ascontiguousarray(MT[ik][act[ik]]), I, J))
        self._packed = packed
        self._pack_tol = tol

    def evaluate(self, E, bias=0.0, tol=1e-14, ik=0, hermitian_parts=False, noopt=True):
        """E:    Complex (or real) (n,) np.ndarray of any length, including 1.
                  The basis table is rebuilt from E on every call, so E may
                  vary freely between calls.
           bias: float, shifting E
           tol:  sparsity tolerance; a value different from the packing
                 tolerance triggers a one-time repack.
           ik:   unused, kept for signature compatibility with the original.
        """
        Ci, Ei, Gi = self.Coeffs, self.Ei, self.Gi
        if hermitian_parts:
            raise NotImplementedError(
                'hermitian_parts=True requires evaluate_*_hermitian from '
                'Block_matrices.Croy, which is not part of this rewrite.')
        if noopt:
            im = -0.5j * evaluate_Lorentz_basis_matrix(Ci, E - bias, Ei, Gi, tol=tol)
            re = 0.5 * evaluate_KK_matrix(Ci, E - bias, Ei, Gi, tol=tol)
            return re + im
        if tol != self._pack_tol:
            self._pack(tol)
        res = np.zeros((Ci.shape[0], len(E), Ci.shape[2], Ci.shape[3]),
                       dtype=np.complex128)
        for ik_idx, (Gpack, I, J) in enumerate(self._packed):
            B = build_B(E - bias, Ei, Gi, ik_idx)
            evaluate_L_KK_table(Gpack, I, J, B, res, ik_idx)
        return res

    def evaluate_gamma(self, E, bias=0.0, tol=1e-15, force_hermitian=False):
        if force_hermitian == False:
            return evaluate_Lorentz_basis_matrix(self.Coeffs, E - bias, self.Ei, self.Gi, tol=tol)
        else:
            raise NotImplementedError(
                'force_hermitian=True requires evaluate_Lorentz_basis_matrix_hermitian '
                'from Block_matrices.Croy, which is not part of this rewrite.')


# ---------------------------------------------------------------------------
# Construction helpers (from the original module)
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True)
def from_eigendecomp(vals, vecs, ivecs, out):
    no = len(vals)
    for io in range(no):
        out += vals[io] * vecs[io, :].reshape(-1, 1) * ivecs[io, :].reshape(1, -1)



def from_saved_file(directory, ik=None, print_nonherm_warning=True):
    """directory: A saved directory containing the needed quantities for the
    propagation scheme, plus the eigenvalues and eigenvectors of the
    coefficient matrix. Returns instances of Lorentzian_SE for each lead."""
    try:
        np.load(directory + '/Superconductor.npy')
        mode = 'direct'
    except:
        mode = 'fromeig'

    if mode == 'fromeig':
        Nl = np.load(directory + '/num_lorentzians.npy')
        Nlead = np.load(directory + '/num_leads.npy')
        xi = flexload(directory + '/xi.npy')
        try:
            Ixi = flexload(directory + '/Ixi.npy')
        except:
            # Added 17.06.2026 to enable smaller file sizes.
            print("Warning: Failed to find Ixi.npy, defaulting to hermitian conjugate of xi")
            Ixi = xi.conj()
        EigVal_Gl = np.load(directory + '/_Gl_Eigenvalues.npy')
        SEs = []
        nk = xi.shape[0]
        if ik is None:
            kidx = np.arange(nk)
        else:
            kidx = np.array([ik])
        for i in range(Nlead):
            Ei = np.load(directory + '/Centres_Lorentzian_' + str(i) + '.npy')
            Gi = np.load(directory + '/Broadening_Lorentzian_' + str(i) + '.npy')
            eigval = EigVal_Gl[i, kidx, :, :]
            eigvec = xi[kidx, i, 0:Nl, :, :]
            ieigvec = Ixi[kidx, i, 0:Nl, :, :]
            if np.allclose(ieigvec.conj(), eigvec) == False:
                print('WARNING: YOUR LOADED EIGENVECTORS ARE NOT RELATED BY A COMPLEX CONJUGATION. PLEASE FIND OUT WHY.')
            no = xi.shape[-1]
            Coeffs = np.zeros((len(kidx), Nl, no, no), dtype=np.complex128)
            for jk in range(len(kidx)):
                for jl in range(Nl):
                    from_eigendecomp(eigval[jk, jl],
                                     eigvec[jk, jl],
                                     ieigvec[jk, jl],
                                     Coeffs[jk, jl])
            SEs += [Lorentzian_SE(Ei, Gi, Coeffs)]
        return SEs
    elif mode == 'direct':
        Nlead = np.load(directory + '/num_leads.npy')
        SEs = []
        for i in range(Nlead):
            Ei = np.load(directory + '/Centres_Lorentzian_' + str(i) + '.npy')
            Gi = np.load(directory + '/Broadening_Lorentzian_' + str(i) + '.npy')
            Coeffs = np.load(directory + '/Gamma_Coeffs.npz')['arr_0'][i]
            SEs += [Lorentzian_SE(Ei, Gi, Coeffs)]
        return SEs
