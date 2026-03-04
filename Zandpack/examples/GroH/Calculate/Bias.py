import numpy as np
from numba import njit
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Initial import name
from Zandpack.Pulses import box_pulse
U       = 3.0
delta_H = 0.5
#@njit
def dH(t,sig):
    no = sig.shape[-1]
    dH = np.zeros((2, no, no), dtype=np.complex128)
    dH[0,84,84]   = U*sig[1,84,84]
    dH[1,84,84]   = U*sig[0,84,84]
    #dH[:,84,84]  += delta_H
    return dH

@njit
def bias(t,a):
    if a == 0:
        return .0#-AP(t)
    if a == 1:
        return .0#+AP(t) #+ .5

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
