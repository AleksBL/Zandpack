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
from Zandpack.Help import TDHelper
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    mu = np.load('StSt.npz')['mud']
    Dev  = load(open('Dev.SiP','rb'))
    k    = np.array((.0, .0, .0))
    def updateH():
        Dev.run_dftb_in_dir(silent= True,subprocess=True,wait=True)
    def evaluateHk():
        return Dev.fast_dftb_hk(k = k, gamma_only = True)
    def q2Q(q):
        Qv[piv] = q[:]
        return Qv
    def updateQ(Qv):
        Q.set_Q(Qv)
    
    piv   = np.load(name+'/Arrays/pivot.npy')
    R     = np.load(name+'/Arrays/Positions.npy')
    Hsisl = Dev.dftb2sisl()
    i2a   = np.array([Hsisl.o2a(i) for i in piv])
    R     = R[i2a]
    Rx    = R[:,0]
    Cx    = np.average(R[:,0])
    R     = Rx[i2a]
    Rmax = Rx.max()
    Rmin = Rx.min()
    Rmp   = (Rx - Cx) /(Rmax - Rmin)
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

