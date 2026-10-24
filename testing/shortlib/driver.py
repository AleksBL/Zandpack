#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shortlib.driver -- the one place a test decides *how* to propagate in Python.

Every test's Python driver calls ``propagate(...)`` here instead of
``Zandpack.decoupled_functions.scipy_ode`` directly, so the choice lives in one
file and the tests stay short.

Environment
    ZP_RHS           'shortlib' (default) | 'make_f'
                     shortlib = the re-implementation in shortlib/rhs.py,
                     make_f   = the package's ``TD_Transport.make_f``.
                     The switch back to the package is one word.
    ZP_CHUNK         eval points integrated per ``solve_ivp`` call (default 1).
                     > 1 uses the piecewise method (shortlib/piecewise.py),
                     which carries the state across pieces and keeps only the
                     current, bounding the solver's peak memory -- see
                     shortlib/README.md for the measured effect.
    ZP_TOL_TIGHTEN   factor the solver tolerances are scaled by when chunking
                     (default 1e-2).  Chunking trades a little accuracy for
                     memory, so the tolerances are tightened to compensate; set
                     it to 1 to disable.

``propagate`` returns ``(t_eval, rho, cl)`` with the same meaning as
``scipy_ode``: ``rho`` the k=0 density matrix at each eval point, ``cl`` the
left current with one column per k-point.
"""

import os

import numpy as np

from shortlib.piecewise import solve_pieces
from Zandpack.decoupled_functions import scipy_ode

__all__ = ['build_f', 'propagate']


def build_f(C, rhs=None):
    """The right-hand side to propagate with, selected by ``ZP_RHS``."""
    rhs = rhs or os.environ.get('ZP_RHS', 'shortlib')
    if rhs == 'shortlib':
        from shortlib.rhs import arrays_from_td, make_f_fast
        from Zandpack.td_constants import hbar
        return make_f_fast(arrays_from_td(C), hbar=float(hbar))
    if rhs == 'make_f':
        return C.make_f()
    raise ValueError("ZP_RHS must be 'shortlib' or 'make_f', not %r" % rhs)


def propagate(C, dH, delta, t0, t_eval, atol, rtol, f=None):
    """Propagate the device ``C`` in Python; drop-in for ``scipy_ode``.

    ``t_eval`` must lie strictly to the right of ``t0`` (as ``scipy_ode``
    asserts).  Reads ``ZP_RHS`` / ``ZP_CHUNK`` / ``ZP_TOL_TIGHTEN``.
    """
    chunk = int(os.environ.get('ZP_CHUNK', '1'))
    if chunk > 1:
        tighten = float(os.environ.get('ZP_TOL_TIGHTEN', '1e-2'))
        atol, rtol = atol * tighten, rtol * tighten
    if f is None:
        f = build_f(C)
    t_eval = np.asarray(t_eval, dtype=float)

    if chunk > 1:
        return solve_pieces(f, C.sigma, C.Psi_vec, C.omegas, t0, t_eval,
                            dH, delta, C.Ixi, chunk=chunk, atol=atol, rtol=rtol)
    return scipy_ode(f, C.sigma, C.Psi_vec, C.omegas, t0, t_eval,
                     dH, delta, C.Ixi, dH_given=True, method='RK45',
                     dt_guess=None, atol=atol, rtol=rtol)
