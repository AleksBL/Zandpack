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


if rank == 0:
    # In the end, the zand code is going to
    # load the Bias and Initial scripts
    # in each process, so we save some memory by
    # only loading things in rank=0.
    Hlp = TDHelper(name)
    S   = Hlp.S
    dm_eq  =  np.load(name+'/Arrays/DM_Ortho.npy')


def q_to_Hno(q):
    return getH(q)

def dH(t,sigma):
    dm_no = Hlp.lowdin_transform(sigma) #L@sigma@L
    # q = mulliken charge on each site in p_z system.
    q     = sig2mul(sigma)# np.diag((dm_no@S + S@dm_no )[0]).real
    V     = bias(t,1) - bias(t,0)
    Hext  = np.diag(Rmp*V)
    H_dm  = q_to_Hno(q)
    dhk   = Hlp.lowdin_transform(H_dm + Hext) - Hlp.bare_H0(orthogonal=True)
    DynCor= Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) 
                                              for a in range(2)
                                              ])
    return dhk + DynCor

def bias(t,a):
    w = 0.5
    A = 0.5
    s   = 20.0**2
    V = A*np.sin(w*t)*np.exp(-(t**2)/s)
    if a == 0: return +V
    if a == 1: return -V


def dissipator(t,sig):
    return 0.0

def sig2mul(sig):
    dm_eqno = Hlp.lowdin_transform(sig)
    Q       = np.diag((dm_eqno@S + S@dm_eqno)[0]).real
    return Q

if linearize and rank == 0:
    Q0      = sig2mul(dm_eq)
    dftb_h0 = getH(Q0)
    dq      = 0.05
    try:
        dHdQ    = np.load('dHdQ.npz')['arr_0']
        dftb_h0 = np.load('dftb_h0.npz')['arr_0']
        Q0      = np.load('Q0.npz')['arr_0']
    except:
        dHdQ = []
        for i in tqdm(range(dm_eq.shape[-1])):
            Qv     = Q0.copy()
            Qv[i] += dq
            dHdQ  += [(getH(Qv) - dftb_h0)/dq]
        dHdQ = np.array(dHdQ).transpose(1,2,0).copy()
        np.savez_compressed('dHdQ.npz', dHdQ)
        np.savez_compressed('dftb_h0.npz', dftb_h0)
        np.savez_compressed("Q0.npz", Q0)
    def q_to_Hno(q):
        dq = q - Q0
        return dftb_h0 + dHdQ @ dq  
         
    def dH(t,sigma):
        dQ  = sig2mul(sigma) - Q0
        V     = bias(t,1) - bias(t,0)
        Hext  =  Hlp.lowdin_transform(np.diag(Rmp*V))
        H_dm  = dftb_h0 + dHdQ@dQ
        DynCor= Hlp.lead_dev_dyncorr(DeltaList=[bias(t, 0), bias(t, 1)])
        dH0   = Hlp.lowdin_transform( H_dm ) - Hlp.bare_H0(orthogonal=True)
        return dH0 + DynCor + Hext
