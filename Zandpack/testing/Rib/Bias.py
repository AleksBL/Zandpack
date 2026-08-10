from numba import njit
import numpy as np
from Zandpack.Pulses import box_pulse

@njit
def bias(t,a):
    if a == 0:
        return .0#-0.5 + (np.cos(t) + np.cos(1.5*t)) * box_pulse(t+20,40,.5, 1.0)
    if a == 1:
        return .0#0.5 - (np.cos(t/3) + np.cos(t)) * box_pulse(t+20,40,.5, 1.0)

@njit
def dH(t,sig):
    o  = np.zeros(sig.shape, dtype=np.complex128)
    #o[0,20,20] += 2.0*o[0,20,20]
    return o


def dissipator(t,s): return 0.0

