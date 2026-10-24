#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shortlib.rhs -- an independent, numba-optimised re-implementation of
``TD_Transport.make_f()``, i.e. the right-hand side of the Zandpack propagation.

The equations it implements, from ``Zandpack.equations``, with the code names
``make_f`` uses:

    device_density_matrix_eom
        i hbar dsigma/dt = [H_D(t), sigma] + i sum_a (Pi_a + Pi_a^dag)

    auxiliary_mode_eom
        i hbar dPsi_axc/dt = [H(t) - I chi^+_ax] Psi_axc
                           + hbar Lam^{<,+}_axc xi_axc
                           + hbar (Lam^{>,+} - Lam^{<,+})_axc sigma xi_axc
                           + (1/hbar) sum_{a'x'c'} Om_axc,a'x'c' xi_a'x'c'

    omega_eom
        i hbar dOm_axc,a'x'c'/dt
              = (chi^-_{a'x'} - chi^+_{ax}) Om
              + i hbar (Lam^{>,-} - Lam^{<,-})_axc xi^dag_{a'x'c'} Psi_axc
              + i hbar (Lam^{>,+} - Lam^{<,+})_axc Psi^dag_{a'x'c'} xi_axc

    equation symbol            code name (in make_f)
    -------------------------  -------------------------------------------
    sigma                      old_sig          (nk, no, no)
    H_D(t)                     Hdense[:,0] + dH(t, sigma)
    Pi_a                       PI_opti(psi[:,a], Ixi[:,a], psi_idx[a])
    Psi_axc                    old_psi          (nk, nlead, Ntot, no, no)
    Om_axc,a'x'c'              old_omega        (nk, nz)   -- packed, see below
    chi^+_ax / chi^-_ax        Xpp / Xpm        (+ delta_a)
    Lam^{<,+} / Lam^{<,-}      GL_P / GL_M
    Lam^{>,+} - Lam^{<,+}      diff_GGP_GLP     (GG_P - GL_P)
    Lam^{>,-} - Lam^{<,-}      diff_GGM_GLM     (GG_M - GL_M)
    xi_axc / xi^dag_axc        xi / Ixi         (nk, nlead, Ntot, no, no)

with ``Ntot = NumL + 2*num_poles`` (Lorentzians plus Fermi poles of both signs)
and ``n_mode = nlead*Ntot*no``, the flat mode index ``a*Ntot*no + x*no + c``.

---------------------------------------------------------------------------
Everything that can be is a matrix product

``make_f`` walks index triple loops with a copy/mask per mode; here the same
arithmetic is written as products over the two flat ``(n_mode, no)`` matrices
``Psi_k`` and ``Xi_k``:

    Pi        = (Psi_k masked).T @ Ixi_k        (no, no)
    H . Psi   = Psi_k @ Ht.T                    (n_mode, no)
    sigma.xi  = Xi_k @ sigma.T                  (n_mode, no)
    Om . xi   = Om_dense @ Xi_k                 (n_mode, no)
    S1        = Psi_k @ Ixi_k.T                 (n_mode, n_mode)
    S2        = Xi_k @ conj(Psi_k).T            (n_mode, n_mode)

The diagonal shift ``[H - I chi] Psi`` is applied as ``Psi @ Ht.T - chi*Psi``
instead of copying a matrix per mode, and nothing is allocated inside a mode
loop.  The index maps are read once as data; no structural assumption is made
about them (every row is visited, and check below), so a fully dense map costs
what it costs -- ``verify_rhs.py`` prints each device's fill fraction.

Which numpy product, and why (the semantics differ in ways that bite):

    np.matmul(A, B) / A @ B    matrix product, no conjugation  <- what is used
    np.matvec(A, v)            = A @ v        (numpy >= 2.2; no conjugation)
    np.vecdot(a, b)            = sum(conj(a) * b)   (conjugated FIRST argument)
    np.vdot(a, b)              same, but *flattens* -- only usable for genuine
                               1-D vectors, of which the RHS has none
    np.vecmat(a, B)            = sum_i conj(a_i) * B[i,:]  (also conjugates, so
                               not interchangeable with a @ B)

The first three of those would say what the per-row products mean more directly
than ``@`` does, but **numba only accepts them when the arrays are function
arguments**: used on a closed-over array (or a module global) it reports "Use of
unsupported NumPy function 'numpy.matvec'".  ``f``'s signature is fixed by
``scipy_ode``/``AdaptiveRK4``, and every array here is necessarily closed over
(exactly as in ``make_f``), so ``@`` / ``np.matmul`` is what can be used.  That
is the only reason the ``conj`` in the ``s2`` line is written explicitly.

Two back ends, because the trade-off is real
--------------------------------------------

``dense_omega=False`` (**the default**) keeps Omega packed and contracts it
entry by entry (still with ``@``, one matvec per mode row), never materialising
``(n_mode, n_mode)``.  ``dense_omega=True`` instead builds Omega as a dense
``(n_mode, n_mode)`` matrix once per k and gets every Omega-dependent term from
the products above; it stays available because on small ``n_mode`` it is the
fastest of the three.

Measured on the three verification devices (per RHS call, verify_rhs.py):

    device              n_mode   fill     make_f   dense=True    dense=False
    synth_dm                24   30.6 %   0.05 ms  0.02 ms 3.2x  0.04 ms 1.4x
    SR                    2088   36.0 % 389.3 ms 263.4 ms 1.5x  224.5 ms 1.7x
    scipy_ode_vs_zand      840    7.5 %  34.6 ms  57.3 ms 0.6x   19.2 ms 1.8x

So the dense back end wins only when ``n_mode`` is small (there the matmuls stop
paying any packing overhead at all); it loses badly when the map is sparse,
where ``n_mode**2`` throws away an order of magnitude more entries than exist.
It is also the risky one for large systems, since it allocates four
``(n_mode, n_mode)`` arrays per call.  ``dense_omega=False`` is never slower, so
it is the default; ``dense_omega=True`` remains selectable.

``parallel`` is left False on purpose: numba's parfors pass cannot type-check
this body (it raises "Sizes of Hpsi, Xp, psi_k do not match" on the row scaling,
and in one arrangement it compiled but silently returned a wrong ``D_psi``).
The prange is over k-points, which is 1-3 in every shipped test, while the time
goes on the products -- and those are BLAS calls either way.

One quirk is reproduced faithfully rather than "fixed": ``PI_opti`` already
divides by ``hbar`` and then ``D_sig *= 1/hbar`` divides the whole right-hand
side again, so the ``Pi`` term carries ``1/hbar**2`` relative to the commutator.
Every test runs with ``hbar = 1``, where that is invisible; running the verifier
with ``ZANDPACK_HBAR != 1`` confirms the placement matches the package.
"""

import numpy as np
from numba import njit, prange

__all__ = ['arrays_from_td', 'make_f_fast']


# ---------------------------------------------------------------------------
# 1. collect the time-independent arrays
# ---------------------------------------------------------------------------
def arrays_from_td(C):
    """Collect everything ``make_f`` closes over, as a plain dict of arrays.

    Explicit rather than "whatever make_f happens to use", because this is also
    the interface a caller can fill from ``<name>/Arrays/`` instead of from a
    pickled ``TD_Transport``.
    """
    H = np.ascontiguousarray(C.Hdense[:, 0, :, :], dtype=np.complex128)
    nk, no = H.shape[0], H.shape[2]
    nl = int(C.num_leads)
    Ntot = int(C.NumL) + 2 * int(C.num_poles)

    A = dict(
        H=H,
        Xpp=np.ascontiguousarray(C.Xpp, dtype=np.complex128),
        Xpm=np.ascontiguousarray(C.Xpm, dtype=np.complex128),
        GL_P=np.ascontiguousarray(C.GL_P, dtype=np.complex128),
        GL_M=np.ascontiguousarray(C.GL_M, dtype=np.complex128),
        dGP=np.ascontiguousarray(C.GG_P - C.GL_P, dtype=np.complex128),
        dGM=np.ascontiguousarray(C.GG_M - C.GL_M, dtype=np.complex128),
        xi=np.ascontiguousarray(C.xi, dtype=np.complex128),
        Ixi=np.ascontiguousarray(C.Ixi, dtype=np.complex128),
        nk=nk, no=no, nl=nl, Ntot=Ntot, nz=int(C.omegas.shape[-1]),
    )
    A['maps'] = index_maps(C.omega_idx, C.psi_idx, Ntot, nl, no)
    return A


def index_maps(omega_idx, psi_idx, Ntot, nl, no):
    """Flatten the two static index maps once (pure data, read not exploited).

    ``omega_idx[a,x,c,a2,x2,c2]`` is the slot of that pair in the packed omega
    vector, or ``-1`` when the pair does not exist; ``psi_idx[a,x,c] >= 0`` marks
    the modes that contribute to ``D_psi``.  Returns arrays over the flat mode
    index ``m = a*Ntot*no + x*no + c``:

        om_row       (n_om,)     flat row mode of each stored entry
        om_src       (n_om,)     its flat source mode
        om_pack      (n_om,)     its slot in the packed omega vector
        om_ptr       (n_mode+1,) row pointers into the three arrays above
        psi_mask     (n_mode,)   bool, psi_idx >= 0
        lead_of_mode (n_mode,)   a of each flat mode (for delta_a)
    """
    n_mode = nl * Ntot * no
    oi = np.ascontiguousarray(omega_idx, dtype=np.int64).reshape(n_mode, n_mode)
    stored = oi >= 0
    om_row, om_src = np.nonzero(stored)

    om_ptr = np.zeros(n_mode + 1, dtype=np.int64)
    np.cumsum(stored.sum(axis=1), out=om_ptr[1:])

    return dict(
        om_row=om_row.astype(np.int64),
        om_src=om_src.astype(np.int64),
        om_pack=oi[om_row, om_src].astype(np.int64),
        om_ptr=om_ptr,
        psi_mask=(np.ascontiguousarray(psi_idx, dtype=np.int64).reshape(n_mode) >= 0),
        lead_of_mode=(np.arange(n_mode, dtype=np.int64) // (Ntot * no)),
    )


# ---------------------------------------------------------------------------
# 2. the right-hand side
# ---------------------------------------------------------------------------
def make_f_fast(A, hbar=1.0, dense_omega=False, parallel=False, fastmath=True):
    """Return a numba function with ``make_f``'s signature:

        f(t, old_sig, old_psi, old_omega, dH, delta_variant, dH_given=True)
            -> (D_sig, D_psi, D_omega)

    See the module docstring for what the two ``dense_omega`` back ends cost.
    """
    H = A['H']
    Xpp, Xpm = A['Xpp'], A['Xpm']
    GL_P, dGP, dGM = A['GL_P'], A['dGP'], A['dGM']
    xi, Ixi = A['xi'], A['Ixi']
    mp = A['maps']
    nk, no, nl, Ntot = A['nk'], A['no'], A['nl'], A['Ntot']
    n_mode = nl * Ntot * no

    om_row, om_src, om_pack = mp['om_row'], mp['om_src'], mp['om_pack']
    om_ptr = mp['om_ptr']
    psi_mask, lead_of_mode = mp['psi_mask'], mp['lead_of_mode']
    n_om = om_pack.shape[0]
    zero = 0.0 + 0.0j

    @njit(parallel=parallel, fastmath=fastmath)
    def f(t, old_sig, old_psi, old_omega, dH, delta_variant, dH_given=True):
        D_sig = np.zeros(old_sig.shape, dtype=np.complex128)
        D_psi = np.zeros(old_psi.shape, dtype=np.complex128)
        D_omega = np.zeros(old_omega.shape, dtype=np.complex128)
        D_psi_f = D_psi.reshape(nk, n_mode, no)     # the flat view used below

        delta_t = np.zeros(nl)
        for a in range(nl):
            delta_t[a] = delta_variant(t, a)

        if dH_given:
            Ht = H + dH(t, old_sig)
        else:
            Ht = dH(t, old_sig)

        for k in prange(nk):
            psi_k = old_psi[k].reshape(n_mode, no)
            xi_k = xi[k].reshape(n_mode, no)
            Ixi_k = Ixi[k].reshape(n_mode, no)
            Xpp_r = Xpp[k].reshape(n_mode)
            Xpm_r = Xpm[k].reshape(n_mode)
            GL_P_r = GL_P[k].reshape(n_mode)
            dGP_r = dGP[k].reshape(n_mode)
            dGM_r = dGM[k].reshape(n_mode)
            chi_p = Xpp_r + delta_t[lead_of_mode]
            chi_m = Xpm_r + delta_t[lead_of_mode]
            # conj(psi) is needed once per omega entry below; hoisting it out of
            # the row loop replaces one (n_r, no) conj+allocation per row with a
            # single (n_mode, no) one per k
            psi_conj = np.conj(psi_k)

            # ---- device density matrix ------------------------------------
            # i hbar dsig/dt = [H,sig] + i sum_a (Pi_a + Pi_a^dag)
            # summing over leads == summing over every psi-active mode, so all
            # of Pi is one masked product rather than one product per lead
            Pi = np.where(psi_mask[:, None], psi_k, zero).T @ Ixi_k
            D_sig[k] = (-1j * (Ht[k] @ old_sig[k] - old_sig[k] @ Ht[k])
                        + (Pi + np.conj(Pi).T) / hbar) / hbar

            if dense_omega:
                # ---- Omega as a dense matrix (a permutation: data movement) -
                Om = np.zeros((n_mode, n_mode), dtype=np.complex128)
                for n in range(n_om):
                    Om[om_row[n], om_src[n]] = old_omega[k, om_pack[n]]

                # ---- auxiliary modes --------------------------------------
                # i hbar dPsi/dt = [H - I chi^+] Psi + hbar Lam^{<,+} xi
                #                  + hbar dLam^+ sigma xi + (1/hbar) Om xi
                # (separate statements: numba will not unify a longer chain of
                #  broadcasts in a single expression)
                Hpsi = psi_k @ Ht[k].T
                sig_xi = xi_k @ old_sig[k].T
                Dpsi = (Hpsi - np.expand_dims(chi_p, 1) * psi_k) / hbar
                Dpsi += np.expand_dims(GL_P_r, 1) * xi_k
                Dpsi += np.expand_dims(dGP_r, 1) * sig_xi
                Dpsi += (Om @ xi_k) / (hbar * hbar)
                Dpsi *= -1j
                D_psi_f[k] = np.where(psi_mask[:, None], Dpsi, zero)

                # ---- omega ------------------------------------------------
                # i hbar dOm/dt = (chi^-_{a'x'} - chi^+_{ax}) Om
                #                 + i hbar dLam^- xi^dag_{a'x'} Psi
                #                 + i hbar dLam^+ Psi^dag_{a'x'} xi
                S1 = psi_k @ Ixi_k.T          # S1[r,s] = Ixi_s . Psi_r
                S2 = xi_k @ psi_conj.T        # S2[r,s] = Psi_s^* . xi_r
                dOm = -1j * (chi_m - np.expand_dims(chi_p, 1)) * Om / hbar
                dOm += np.expand_dims(dGM_r, 0) * S1
                dOm += np.expand_dims(dGP_r, 1) * S2
                for n in range(n_om):
                    D_omega[k, om_pack[n]] = dOm[om_row[n], om_src[n]]
            else:
                # ---- same arithmetic, one mode row at a time --------------
                # every row is visited; Omega stays packed, so no
                # (n_mode, n_mode) is ever formed
                for row in range(n_mode):
                    p1, p2 = om_ptr[row], om_ptr[row + 1]
                    a = lead_of_mode[row]
                    psi_r = psi_k[row]
                    xi_r = xi_k[row]

                    if psi_mask[row]:
                        Hpsi_r = Ht[k] @ psi_r
                        sig_xi_r = old_sig[k] @ xi_r
                        om_r = np.zeros(no, dtype=np.complex128)
                        if p2 > p1:
                            om_r = (old_omega[k, om_pack[p1:p2]]
                                    @ xi_k[om_src[p1:p2]])
                        dpsi_r = (Hpsi_r - chi_p[row] * psi_r) / hbar
                        dpsi_r += GL_P_r[row] * xi_r
                        dpsi_r += dGP_r[row] * sig_xi_r
                        dpsi_r += om_r / (hbar * hbar)
                        D_psi_f[k, row] = -1j * dpsi_r

                    if p2 > p1:
                        pack = om_pack[p1:p2]
                        src = om_src[p1:p2]
                        w = old_omega[k, pack]
                        chi_m_src = chi_m[src]
                        s1 = Ixi_k[src] @ psi_r
                        s2 = psi_conj[src] @ xi_r
                        D_omega[k, pack] = (-1j * (chi_m_src - chi_p[row]) * w / hbar
                                            + dGM_r[src] * s1
                                            + dGP_r[row] * s2)
        return D_sig, D_psi, D_omega

    return f
