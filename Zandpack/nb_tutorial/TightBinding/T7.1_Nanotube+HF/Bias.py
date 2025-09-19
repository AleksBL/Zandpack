import numpy as np
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Zandpack.Pulses import box_pulse

def bias(t,a):
    if   a == 0: return +box_pulse(t-50, 30, 2.0, 2.0)
    elif a == 1: return -box_pulse(t-50, 30, 2.0, 2.0)
    

def dH(t, sigma):
    A = np.zeros(sigma.shape,dtype = np.complex128)
    return A


def dissipator(t,sig):
    return 0.0

