import numpy as np
from numba import njit
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Initial import name
from Zandpack.Pulses import box_pulse

U  = 3.0
n  = np.load('../spin-density.npy')
p  = np.load('U_zgnr/Arrays/pivot.npy')
n  = n[:,p]

@njit
def dH(t,sig):
    no = sig.shape[-1]
    dH = np.zeros((2, no, no), dtype=np.complex128)
    for i in range(no):
        dH[0,i,i] = U*(sig[1,i,i] - n[1,i]) 
        dH[1,i,i] = U*(sig[0,i,i] - n[0,i])
    return dH

@njit
def bias(t,a):
    if a == 0:
        return -AP(t)#-box_pulse(t-80,50.0, 5.0, 1.0)#1*AP(1*t) 
    if a == 1:
        return +AP(t)#+box_pulse(t-80,50.0, 5.0, 1.0)#1*AP(1*t)

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
