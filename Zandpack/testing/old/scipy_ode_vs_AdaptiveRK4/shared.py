import numpy as np
from numba import njit

@njit
def dH(t,sigma):
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    out[0, 0, 0] = np.sin(t/10)
    out[0, 5, 5] = np.sin(t/6)
    out[0,13,13] = np.sin(t/2)
    return out

@njit
def delta(t,a):
    if a == 0:
        return +np.sin(t/20)
    return -np.sin(t/20)
