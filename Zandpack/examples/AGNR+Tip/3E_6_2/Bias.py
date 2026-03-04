import numpy as np
from numba import njit
from Zandpack.Interpolation import make_spline
from Zandpack.Pulses import air_photonics_pulse as AP
from Initial import name
from DFTB_driver import getH, q0, Rmp
from Zandpack.Pulses import box_pulse
from Zandpack.Pulses import  air_photonics_pulse as AP
from Zandpack.Help import TDHelper
from mpi4py import MPI
from tqdm import tqdm
import os

linearize = True if os.environ['LINEARIZE']=='True' else False
rank      = MPI.COMM_WORLD.Get_rank()

Hlp = TDHelper('3E_new')
if rank == 0:
    S   = Hlp.S

def q_to_Hno(q):
    return getH(q)

def dH(t,sigma):
    dm_no = Hlp.lowdin_transform(sigma) #L@sigma@L
    # q = mulliken charge on each site in p_z system.
    q     = np.diag((dm_no@S + S@dm_no )[0]).real
    V     = bias(t,2)
    Hext  = np.diag(Rmp*V)
    dhk   = Hlp.lowdin_transform(q_to_Hno(q) + Hext) - Hlp.bare_H0(orthogonal=True)
    DynCor= Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) 
                                              for a in range(3)
                                              ])
    return dhk + DynCor

@njit
def bias(t,a):
    if a == 0: return 0.0
    if a == 1: return 0.0
    if a == 2: return AP(t)

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

def sig2mul(sig):
    dm_eqno = Hlp.lowdin_transform(sig)
    Q       = np.diag((dm_eqno@S + S@dm_eqno)[0]).real
    return Q

if linearize and rank == 0:
    Q0      = sig2mul(dm_eq)
    dftb_h0 = getH(Q0)
    dq      = 0.05
    try:
        dHdQ = np.load('dHdQ.npz')['arr_0']
        dftb_h0 = np.load('dftb_h0.npz')['arr_0']
    except:
        dHdQ = []
        for i in tqdm(range(dm_eq.shape[-1])):
            Qv     = Q0.copy()
            Qv[i] += dq
            dHdQ  += [(getH(Qv) - dftb_h0)/dq]
        dHdQ = np.array(dHdQ).transpose(1,2,0).copy()
        np.savez_compressed('dHdQ.npz', dHdQ)
        np.savez_compressed('dftb_h0.npz', dftb_h0)
    
    def q_to_Hno(q):
        dq = q - Q0
        return dftb_h0 + dHdQ @ dq  
         
    #def dH(t,sigma):
    #    dQ  = sig2mul(sigma) - Q0
    #    res = np.zeros(Helper.H0.shape,dtype=complex)
    #    nk  = Helper.H0.shape[0]
    #    for ik in range(nk):
    #        res[ik] = dHdQ[ik,:] @ np.diag(dQ[ik])
    #    # This should always be here
    #    res += Helper.lead_dev_dyncorr(DeltaList=[bias(t, 0), bias(t, 1)])
    #    return dH0 + res
