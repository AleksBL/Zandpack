import numpy as np
from numba import njit

@njit
def dH(t, sig):
    o = np.zeros(sig.shape, dtype = np.complex128)
    o[0,4,4] += sig[0,4,4] - sig[0,3,3]/3
    o[0,3,3] -= sig[0,3,3] + sig[0,4,4]/3
    return o

@njit
def delta(t, a):
    # symmetric Lorentzian pulse centred at t = 75
    if a == 0:
        return +0.05/((t - 75)**2 + 0.025)
    else:
        return -0.05/((t - 75)**2 + 0.025)
