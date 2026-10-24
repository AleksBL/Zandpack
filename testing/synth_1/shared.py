import numpy as np
from numba import njit

@njit
def sawtooth(t, period = 4.0, edge = 0.05):
    """Continuous sawtooth in [-1, 1]: a linear ramp per period whose reset is
    smeared over `edge` (same time units as `period`), so there is no jump in
    the value and no infinite derivative."""
    xi = (t/period) - np.floor(t/period)          # 0 -> 1 ramp
    return 2.0*(xi - 0.5*(1.0 + np.tanh((xi - 1.0)/edge))) - 1.0

@njit
def dH(t, sigma):
    # drive the two end sites of the chain
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    out[0,  0,  0] = np.sin(t/10)
    out[0, -1, -1] = np.sin(t/6)
    return out

@njit
def delta(t, a):
    if a == 0:
        return +np.sin(t/20)
    return -np.sin(t/20)

@njit
def dH_saw(t, sigma):
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    out[0,  0,  0] = sawtooth(t)
    out[0, -1, -1] = sawtooth(t)
    return out

@njit
def delta_saw(t, a):
    if a == 0:
        return +sawtooth(t)
    return -sawtooth(t)
