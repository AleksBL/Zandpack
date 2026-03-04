#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:29:36 2022

@author: aleksander
"""

import numpy as np
import sisl
from time import time,sleep
import os
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester as S_S
from Zandpack.Pulses import air_photonics_pulse as AP
from mpi4py import MPI
from Zandpack.Help import TDHelper
from params import name, oDir, CalcDir
linearize = True if os.environ['linearize']=='True' else False

rank = MPI.COMM_WORLD.Get_rank()
Hlp = TDHelper(name)
dmf0  = sisl.get_sile(oDir+'siesta.TSDE').read_density_matrix()
E_F   = sisl.get_sile(oDir+'RUN.out{stdoutSileSiesta}').read_energy()['fermi']
piv   = np.load(name+'/Arrays/pivot.npy')
devg  = sisl.get_sile(CalcDir+'siesta.XV').read_geometry(); devg.set_nsc((1,1,1))
if rank == 0:
    H_or  = sisl.get_sile(oDir+'siesta.TSHS').read_hamiltonian()
    Z     = H_or.xyz[H_or.o2a(piv)][:, 2].copy()
    z_cent= np.average(Z)
    Rmp   = (Z - z_cent)/(Z.max() - Z.min())
    del H_or
    S   = Hlp.S
    U   = Hlp.Lowdin
    iU  = Hlp.invLowdin

dmf   = dmf0.copy()
dm    = np.load(name+'/Arrays/DM_Ortho.npy')

I, J = [], []
for i in piv:
    for j in piv:
        I+=[i]; J+=[j]
I,J = np.array(I), np.array(J)

def signal_new_dm():
    with open('NEWDM.txt','w') as f:
        f.write('NEW')

def wait_for_new_H():
    cond = True
    while cond:
        f = os.listdir()
        sleep(0.005)
        if 'NEWH.txt' in f:
            cond = False # break loop
    sleep(0.01)
    return

def remove_NEWH():
    os.remove('NEWH.txt')

def dH(t,sigma, return_raw_H = False):
    dsig  =  sigma
    dsigNO = (U@dsig@U)[0].ravel() # Gamma only
    for i in range(len(I)):
        dmf[I[i],J[i],0] = 2*dsigNO[i].real
    dmf.write(CalcDir+'siesta.DM')
    signal_new_dm()
    wait_for_new_H()
    Hload = sisl.get_sile(CalcDir+'siesta.HSX').read_hamiltonian(geometry = devg)
    Hload.set_nsc((1,1,1))
    os.system('rm '+CalcDir+'siesta.HSX')
    remove_NEWH()
    Hload.shift(-E_F)
    HfNO  = Hload.Hk()[:,piv][piv,:].toarray()
    V_t   = bias(t, 1) - bias(t, 0)
    Hext  = np.diag(Rmp*V_t)
    if return_raw_H:
        return Hlp.lowdin_transform(HfNO)
    dhk   = Hlp.lowdin_transform(HfNO + Hext) - Hlp.bare_H0(orthogonal=True)
    DynCor= Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in [0,1]])
    return dhk + DynCor

def bias(t,a):
    V = AP(t)
    if a == 0:
        return -V
    else:
        return V
    
def sig2mul(dm):
    nodm = U@dm@U
    return S@nodm + nodm@S

def mul2sig(Q):
    if len(Q.shape) == 2: out =  S_S(S,S, Q)
    else:                 out =  np.array([S_S( S[i],S[i],Q[i] ) for i in range(len(Q))])
    out = iU@out@iU
    return out

def dissipator(t,sig):
    return 0.0

# assert 1 == 0
# Linearization part
if linearize and rank == 0:
    dq   = 0.05
    try:
        dHdQ = np.load('dHdQ.npz')['arr_0']
        h0_s = np.load('h0_siesta.npy')
        Q0   = np.load('Q0.npy')
        dH0  = np.load('dH0.npy')
    except:
        dHdQ = []
        h0_s = dH(-80.0, dm, return_raw_H=True)
        Q0   = sig2mul(dm)
        dH0  = dH(-80.0,dm)
        for i in range(dm.shape[-1]):
            Qv         = Q0.copy()
            Qv[0,i,i] += dq
            dHdQ      += [(dH(-80.0, mul2sig(Qv)) - dH0)/dq]
        dHdQ = np.array(dHdQ).transpose(1,2,3,0).copy()
        np.savez_compressed('dHdQ.npz', dHdQ)
        np.save('h0_siesta.npy', h0_s)
        np.save('Q0.npy', Q0)
        np.save('dH0.npy', dH0)
    def dH(t,sigma):
        dQ  = sig2mul(sigma) - Q0
        res = np.zeros(dm.shape,dtype=complex)
        nk  = dm.shape[0]
        for ik in range(nk):
            res[ik] = dHdQ[ik,:] @ np.diag(dQ[ik])
        DynCor  = Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in [0,1]])
        dH_elec = dH0 + res
        V_t   = bias(t, 1) - bias(t, 0)
        Hext  = Hlp.lowdin_transform(np.diag(Rmp*V_t))
        return dH_elec + DynCor + Hext
        
        
        
        
