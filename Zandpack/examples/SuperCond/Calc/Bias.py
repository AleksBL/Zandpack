import numpy as np
from numba import njit
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Initial import name
from Zandpack.Pulses import box_pulse
@njit
def dH(t,sig):
    no = sig.shape[-1]
    dH = np.zeros((1, no, no), dtype=np.complex128)
    return dH

@njit
def bias(t,a):
    if a == 0:
        return box_pulse(t+50, 10, 1, 1)
    elif a == 1:
        return -box_pulse(t+50, 10, 1, 1)

@njit
def step10(t, ts, s):
    return 1/(1 + np.exp((t-ts)/s))
dm_eq  =  np.load(name+'/Arrays/DM_Ortho.npy')
eta    =  0.0
@njit
def ETA(t):
    return eta*step10(t, -50, 10)
@njit
def dissipator(t,sig):
    return ETA(t) * (sig - dm_eq)
