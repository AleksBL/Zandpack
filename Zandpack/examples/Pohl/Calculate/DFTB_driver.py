#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:55:50 2023

@author: aleksander
"""

from pickle import load
import numpy as np
from Initial import name
from time import time
from mpi4py import MPI
import sisl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    mum = np.load('CP.npz')['mum']
    mu = +mum +0.15
    Dev  = load(open('Dev.SiP','rb'))
    k    = np.array((.0, .0, .0))
    def updateH():
        Dev.run_dftb_in_dir(silent= True,subprocess=True,wait=True)
    def evaluateHk():
        return Dev.fast_dftb_hk(k = k)
    def q2Q(q):
        Qv[piv] = q[:]
        return Qv
    def updateQ(Qv):
        Q.set_Q(Qv)
    
    piv  = np.load(name+'/Arrays/pivot.npy')
    R    = np.load(name+'/Arrays/Positions.npy')
    L    = np.load(name+'/Arrays/S^(-0.5).npy')
    Hsisl= Dev.dftb2sisl()
    #Hsisl = sisl.get_sile('Hsisl.TSHS').read_hamiltonian()
    i2a  = np.array([Hsisl.o2a(i) for i in piv])
    R    = R[i2a]
    Rxmax= R[:,0].max()
    Rxmin= R[:,0].min()
    Rxc  = np.average(R[R[:,2]<0.5,0])
    Rmp  = 2*(R[:,0] - Rxc)/(Rxmax - Rxmin)
    Rmp[R[:,2]>0.5] = 0.0
    AlPot  = (R[:,2]>0.5).astype(float)
    # Q is instance of charge interface class
    Q    = Dev.dic['dftb_charge']
    # starting charge
    _Q0  = Q.dic['init_Q'].copy()
    # Modifiable Q, getting modified as the calculation runs
    Qv   = _Q0.copy()
    # subbed
    q0   = _Q0[piv]
    # electrode chemical potential
    sk = Dev.fast_dftb_hk(k = k, label = 'overreal')
    def getH(qv):
        updateQ(q2Q(qv))
        updateH()
        hk = evaluateHk() - mu*sk
        return hk[piv,:][:,piv]
else:
    getH = None
    q0   = None
    Rmp  = None

