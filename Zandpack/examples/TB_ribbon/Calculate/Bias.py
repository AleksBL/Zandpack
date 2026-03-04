import numpy as np
from numba import njit

@njit
def dH(t,sigma):
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    out[0, 0, 0] = np.sin(t/10)
    out[0, 5, 5] = np.sin(t/10)
    out[0, 5, 7] = np.sin(t/10)
    out[0, 7, 5] = np.sin(t/10)
    return out

@njit
def bias(t,a):
    if a == 0:
        return +np.sin(t/20)
    return -np.sin(t/20)

def dissipator(t, sigma):
    return 0.0
