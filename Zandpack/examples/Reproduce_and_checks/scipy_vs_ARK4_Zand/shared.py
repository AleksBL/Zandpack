import numpy as np
from numba import njit

@njit
def dH(t,sigma):
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    s = step(t)
    out[0, 0, 0] = s*np.sin(t/10)
    out[0, 5, 5] = s*np.sin(t/6)
    out[2, 3, 3] = s*np.sin(t/2)
    out[1, 9, 9] = s*np.sin(t/2)
    out[1, 5, 9] = s*np.sin(t/2)/2
    out[1, 9, 5] = s*np.sin(t/2)/2
    
    return out
@njit
def step(t):
    return 1/(1 + np.exp((t - 12)/0.1))


@njit
def delta(t,a):
    s = step(t)
    if a == 0:
        return +s*np.sin(t)
    if a == 1:
        return s*(-np.sin(t+np.pi/5) + 5*np.exp(-((t - 3)*5)**2))
    else:
        return s*(1-np.arctan(t))
    


