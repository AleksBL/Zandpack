import numpy as np
from Zandpack.Pulses import zero_dH

Vmax = 1.0
T = 0.1
W = 0.1
Tmax=10*np.pi
zeta = Tmax

def pulse(t):
    return np.cos(W*t)#*np.exp(-1/2*(t/zeta)**2)

def dH(t,sigma):
    A = np.zeros(sigma.shape, dtype = np.complex128)
    A[0,1,1] = 1 * pulse(t)
    return A

def bias(t,a):
    if  a == 0:
        return 0.0 * pulse(t) 
    if  a == 1:
        return 0.125 * pulse(t)
def dissipator(t,s):return 0.0
