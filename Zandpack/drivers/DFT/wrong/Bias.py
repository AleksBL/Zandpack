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

Name  = 'AlJunc_file'
dmf   = sisl.get_sile('../Device/siesta.DM').read_density_matrix()
E_F   = sisl.get_sile('../Device/RUN.out').read_energy()['fermi']
piv   = np.load(Name+'/Arrays/pivot.npy')
H0    = np.load(Name+'/Arrays/H_Ortho.npy')
HRC   = np.load(Name+'/Arrays/Hamiltonian_renormalisation_correction.npy')
devg  = sisl.get_sile('Device.xyz').read_geometry()
U     = np.load(Name+'/Arrays/S^(-0.5).npy')
HRCNO =  np.linalg.inv(U) @ HRC @ np.linalg.inv(U)

I,J = [], []
for i in piv:
    for j in piv:
        I+=[i]
        J+=[j]

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
    return

def remove_NEWH():
    os.remove('NEWH.txt')

def dH(t,sigma):
    sigNO = (U@sigma@U)[0].ravel() # Gamma only
    dmf[I,J,0] = 2*sigNO[:].real         #
    charge = 0.0
    for i in range(dmf.no):
        charge += dmf[i,i,0]
    print('Charge in device: ', charge) 
    dmf.write('Device/siesta.DM')  # 
    signal_new_dm()
    wait_for_new_H()
    Hload = sisl.get_sile('Device/siesta.HSX').read_hamiltonian(geometry = devg)
    #Hload.shift(-E_F)
    HfNO  = Hload.Hk()[:,piv][piv,:].toarray()
    remove_NEWH()
    Hf = U@HfNO@U
    return Hf - M

M    = np.zeros(H0.shape,dtype=complex)
sig0 = np.zeros(H0.shape,dtype=complex)
for i in range(len(piv)):
    for j in range(len(piv)):
        sig0[0,i,j] = dmf[piv[i],piv[j]][0]

sig0 = np.linalg.inv(U) @ sig0 @ np.linalg.inv(U) 
M    = dH(0.0, sig0)

def bias(t,a):
    return 0.0

