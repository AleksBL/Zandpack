"""Drive and dH for the density-matrix-feedback test.

The orthogonal (Lowdin) basis, which is what `make_f`/`zand`/`scipy_ode` and
`AdaptiveRK4` all work in.  This is the file the `zand` run imports as Bias.py.

The mean-field term reads the density matrix, so the RHS is non-linear.  The
`Bias_NO.py` twin has to convert *both* the argument and the result, because
`nozand` hands `dH` the density matrix in the nonorthogonal basis.
"""

import numpy as np
from numba import njit

# ---- tuning block ---------------------------------------------------------
Rep  = 0.5     # mean-field coupling: the potential on a site follows the charge
Vmax = 1.0     # amplitude of the time-dependent drive
# --------------------------------------------------------------------------


@njit
def dH(t, sigma):
    """Drive on the two end sites, plus a cross-charge mean-field feedback.

    `sigma` is the density matrix at the current Runge-Kutta stage, so
    `out[0, i, i] = Rep * sigma[0, j, j]` couples the potential on site `i` to
    the charge on site `j`.  This is the pattern from
    examples/ThesisExample/Bias.py.
    """
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    out[0,  0,  0]  = +Vmax/2 * np.sin(t/10)
    out[0, -1, -1]  = -Vmax/2 * np.sin(t/10)
    out[0,  0,  0] += Rep * sigma[0, -1, -1]
    out[0, -1, -1] += Rep * sigma[0,  0,  0]
    return out


@njit
def delta(t, a):
    if a == 0:
        return +Vmax * np.sin(t/20)
    return -Vmax * np.sin(t/20)
