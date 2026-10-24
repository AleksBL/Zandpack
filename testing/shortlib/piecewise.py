#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shortlib.piecewise -- the scipy_ode propagation driven in pieces.

Why this exists: the package's ``scipy_ode`` integrates the window as **one
solve_ivp call per eval point** (see ``decoupled_functions.propagate``) and keeps
the accumulated current/DM lists; its peak memory grows with that call count --
measured on ``SR`` at a fixed ``t = 0 ... 2``: N = 50 -> >12 GB, N = 10 ->
3.5 GB.  Reducing the number of solver calls therefore reduces the peak, and
that is what ``chunk`` does here: a piece integrates ``chunk`` eval points in one
``solve_ivp`` call and reads the intermediate points from its dense output.  The
state ``(sigma, psi, omega)`` is carried across pieces, so only the current is
kept.

The package's ``scipy_ode`` cannot be used for this because it does not return
the state -- line 697 of ``decoupled_functions.py`` is

    return t_eval, np.array(rho), np.array(cl), #state_sig, state_psi, state_omg

i.e. ``Psi`` and ``Omega`` are commented out (only ``Sigma`` is recoverable, as
``rho[-1]``).  So the loop is re-implemented here, but *only* the loop: the state
packing, the current matrix and the current are the package's own
``one2three``/``three2one``/``PI``/``Jk``.

``chunk = 1`` reproduces the package's call pattern exactly (one solver call per
eval point), so it is the control for the comparison.

Contract: same as ``Zandpack.decoupled_functions.scipy_ode``,
``(t_eval, rho, cl)`` -- ``rho`` the k=0 density matrix at each eval point,
``cl`` the left current with one column per k-point.
"""

import numpy as np
from scipy.integrate import solve_ivp
from Zandpack.decoupled_functions import one2three, three2one, PI, Jk

__all__ = ['solve_pieces']


def solve_pieces(f, sig0, psi0, omg0, t0, t_eval, dH, delta, Ixi,
                 chunk=1, atol=1e-6, rtol=1e-4, method='RK45', report=None):
    """Propagate ``f`` over ``t_eval``, integrating ``chunk`` points per call.

    ``dH`` and ``delta`` are passed straight to ``f`` (``dH_given=True``), as in
    ``scipy_ode``.  ``method`` only documents the solver: the package's
    ``propagate`` never forwards its own ``method`` argument, so ``solve_ivp``'s
    default (RK45) is what both use.
    """
    nk, no = sig0.shape[0], sig0.shape[1]
    noT, nlead, Nm = psi0.shape[3], psi0.shape[1], psi0.shape[2]
    nz = omg0.shape[-1]
    t_eval = np.asarray(t_eval, dtype=float)
    assert (t0 < t_eval).all()

    def F(t, y):
        y1, y2, y3 = one2three(y, nk, no, noT, nlead, Nm, nz)
        d1, d2, d3 = f(t, y1, y2, y3, dH, delta, dH_given=True)
        return three2one(d1, d2, d3)

    sig, psi, omg = sig0.copy(), psi0.copy(), omg0.copy()
    rho, cl = [], []
    t_prev, i, n = float(t0), 0, len(t_eval)

    while i < n:
        j = min(i + chunk, n)                      # eval points in this piece
        sol = solve_ivp(F, (t_prev, float(t_eval[j - 1])),
                        three2one(sig, psi, omg), method=method,
                        atol=atol, rtol=rtol, t_eval=t_eval[i:j])
        for c in range(sol.y.shape[1]):            # one column per eval point
            s, p, _ = one2three(sol.y[:, c], nk, no, noT, nlead, Nm, nz)
            rho.append(s[0])
            cl.append(Jk(PI(p[:, 0], Ixi[:, 0])))
        sig, psi, omg = one2three(sol.y[:, -1], nk, no, noT, nlead, Nm, nz)
        t_prev, i = float(t_eval[j - 1]), j
        if report is not None:
            report(t_prev, len(sol.t))

    return t_eval, np.array(rho), np.array(cl)
