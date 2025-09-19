import numpy as np
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Zandpack.Pulses import box_pulse

s = 5
A = 2.0
w = 0.05
def Pulse(t):
    return A*np.exp(-(t/s)**2)*np.sin(2*np.pi*w*t)

def bias(t,a):
    if a==0: return -Pulse(t)
    if a==1: return +Pulse(t)
X = np.load('Chain/Arrays/Positions.npy')[np.load('Chain/Arrays/pivot.npy'),0] 
# Remember the pivot here because tbtrans may have switched some rows
# the electrode incices has also been removed in pivot
tx = len(X)/2
def dH(t, sig):
    return 0.0
#def dH(t, sig):
#    A  = np.zeros(sig.shape,dtype = np.complex128)
#    xm = tx/2
#    A[0,:,:] = -np.diag(2*(X-xm)/(X.max() - X.min())) * Pulse(t)
#    return A
#def bias(t,a):
#    if a == 0:   return +box_pulse(t-50, 30, 2.0, 2.0)
#    elif a == 1: return -box_pulse(t-50, 30, 2.0, 2.0)

#def dH(t, sigma):
#    A = np.zeros((1,5,5),dtype = np.complex128)
#    return A


def dissipator(t,sig):
    return 0.0

