import numpy as np
from numba import njit
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Initial import name, hbar


dipole_moment = 0.3 * hbar
w12           = 1
Rabi_f        = dipole_moment / hbar
pert_w        = 0.9
@njit
def bias(t,a):
    if a == 0:   return 0.0
    elif a == 1: return 0.0
@njit
def dH(t, sigma):
    A = np.zeros((1,2,2),dtype = np.complex128)
    if t>80.0:
        return A
    if t > 0.0:
        A[0, 0,1]   =  dipole_moment*np.cos(pert_w * t)#  Bias(t, 20, 1)/4 #np.sin(t)#
        A[0, 1,0]   =  dipole_moment*np.cos(pert_w * t)#- Bias(t, 20, 1)/4 #np.sin(t)#
    return A
@njit
def dissipator(t,sig):
    return 0.0

