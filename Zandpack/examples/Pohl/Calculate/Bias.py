import numpy as np
from numba import njit
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Initial import name
from DFTB_driver import getH, q0, Rmp, AlPot
from Zandpack.Pulses import box_pulse

HC = np.load(name+'/Arrays/Hamiltonian_renormalisation_correction.npy')[0]
H0 = np.load(name+'/Arrays/H_Ortho.npy')
L  = np.load(name+'/Arrays/S^(-0.5).npy')
iL = np.linalg.inv(L)
S  = iL@iL
I  = np.eye(288)

def dH(t,sigma):
    dm_no = L@sigma@L
    # q = mulliken charge on each site in p-z system.
    q     = np.diag((dm_no@S + S@dm_no )[0]).real
    #dq    = q - q0
    V     = (bias(t,1) - bias(t,0))*0.5
    Hext  = np.diag(Rmp*V) #+ (-0.0  * AlPot)
    dhk   = L@(getH(q) + Hext)@L - H0 + HC
    return dhk
    
    #return -0.57*I#0.5* AlPot##np.zeros(sigma.shape,dtype=complex)

@njit
def bias(t,a):
    if a == 0:
        return 0.0# - box_pulse(t,50.0, 5.0, 1.0)#1*AP(1*t) 
    if a == 1:
        return 0.0#+ box_pulse(t,50.0, 5.0, 1.0)#1*AP(1*t)
    if a == 2:
        return 0.0

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
