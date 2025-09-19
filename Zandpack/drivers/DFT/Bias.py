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


Name  = 'Benzene_41L'
dmf0   = sisl.get_sile('../Device/siesta.TSDE').read_density_matrix()
E_F   = sisl.get_sile('../Device/RUN.out').read_energy()['fermi']
piv   = np.load(Name+'/Arrays/pivot.npy')
H0    = np.load(Name+'/Arrays/H_Ortho.npy')
HRC   = np.load(Name+'/Arrays/Hamiltonian_renormalisation_correction.npy')
devg  = sisl.get_sile('../Device/siesta.XV').read_geometry()
U     = np.load(Name+'/Arrays/S^(-0.5).npy')
iU    = np.linalg.inv(U)
dmf   = dmf0.copy() 
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
    sleep(0.01)
    return

def remove_NEWH():
    os.remove('NEWH.txt')

sig0 = np.load('out.npy')
def dH(t,sigma):
    dsig  =  sigma#-sig0
    dsigNO = (U@dsig@U)[0].ravel() # Gamma only
    for i in range(len(I)):
        dmf[I[i],J[i],0] = 2*dsigNO[i].real # dmf0[I[i],J[i],0] + 2*dsigNO[i].real         #
    
    dmf.write('Device/siesta.DM')  # 
    signal_new_dm()
    wait_for_new_H()
    Hload = sisl.get_sile('Device/siesta.HSX').read_hamiltonian(geometry = devg)
    os.system('rm Device/siesta.HSX')
    remove_NEWH()
    Hload.shift(-E_F)
    HfNO  = Hload.Hk()[:,piv][piv,:].toarray()
    diag = np.diag(HfNO)
    print(diag.shape)
    plt.scatter(np.arange(len(diag)), diag)
    plt.scatter(np.arange(len(diag)), np.diag((iU@(H0 - HRC[0])@iU)[0]))
    plt.savefig('OS.png')
    plt.close()
    #print('Charge in Device was really ',(dmf.Dk() * Hload.Sk()).sum())
    Hf = U@HfNO@U
    return Hf - M
M = np.zeros(sig0.shape,dtype = complex)
M = dH(0.0, sig0)

def bias(t,a):
    return 0.0

