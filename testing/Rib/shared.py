import numpy as np
from numba import njit
from Zandpack.Pulses import box_pulse

@njit
def dH(t, sig):
    o = np.zeros(sig.shape, dtype = np.complex128)
    return o

@njit
def delta(t, a):
    if a == 0:
        return -0.5 + (np.cos(t) + np.cos(1.5*t)) * box_pulse(t+20, 40, .5, 1.0)
    if a == 1:
        return  0.5 - (np.cos(t/3) + np.cos(t)) * box_pulse(t+20, 40, .5, 1.0)
