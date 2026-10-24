import numpy as np
from numba import njit
from Zandpack.Pulses import box_pulse
Vmax = 1.5
def dH(t,sigma):
    A = np.zeros(sigma.shape, dtype = np.complex128)
    A[0,0,0] = +Vmax/2 * box_pulse(t, 20, 1, 1)
    A[0,1,1] = -Vmax/2 * box_pulse(t, 20, 1, 1)
    return A
def bias(t,a):
    if  a == 0:
        return +Vmax * box_pulse(t, 20, 1, 1)
    if  a == 1:
        return -Vmax * box_pulse(t, 20, 1, 1)
def dissipator(t,sig): return 0.0
