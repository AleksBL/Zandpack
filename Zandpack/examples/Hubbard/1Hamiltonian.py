#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:44:17 2023

@author: aleksander
"""
import sisl
from siesta_python.siesta_python import SiP
import numpy as np
import matplotlib.pyplot as plt
from hubbard import HubbardHamiltonian, NEGF
from ase.visualize import view
tx   = 16
ty   = 11
U    = 3.0
kT   = 0.025
ZGNR = sisl.geom.zgnr(ty)
dev  = ZGNR.tile(tx,0)
emg  = ZGNR.copy()
hem0 = sisl.Hamiltonian(emg)
hem0.construct([[0.1, 1.5], [0.0, -2.7]])
epg  = ZGNR.move(ZGNR.cell[0] * (tx - 1))
if ty == 7 and tx == 10:
    remove =[55, 62, 61, 69, 68, 75, 76, 83, 82, 90, 89, 97,]
elif ty == 11 and tx == 16:
    #remove = [#88, 99, 100, 111,110, 112, 123, 134, 144, 143, 132,133, 121, 122, 154,
              #220,231, 242, 253, 264, 275, 286, 
              #197, 208, 207, 219, 
              #109, 120, 131,
              #
    remove = [131, 153, 175, 197, 142, 164, 186, 141, 163, 185, 152, 174,
              167, 156, 178, 144, 177, 155, 144, 166, 188, 143, 165, 187, 132, 154, 176, 198]
    #remove = []
    

dev  = dev.remove(remove)
#assert 1 == 0
em = SiP(
    emg.cell,
    emg.xyz,
    emg.to.ase().numbers,
    directory_name="EM",
    kp=[50, 1, 1],
    sm="EM",
    sl="EM",
    semi_inf="-a1",
)

em.manual_H(hem0)
nup = np.where(emg.xyz[:, 1] > emg.center()[1])[0]  # + np.random.random(len(ZGNR))
ndn = np.where(emg.xyz[:, 1] < emg.center()[1])[0]  # + np.random.random(len(ZGNR))
MFH_EM = em.Hubbard_electrode(nup, ndn, return_MFH = True, U=U,kT=kT)


ep = SiP(
    epg.cell,
    epg.xyz,
    epg.to.ase().numbers,
    directory_name="EP",
    kp=[50, 1, 1],
    semi_inf="+a1",
    sl="EP",
    sm="EP",
)
hep0 = sisl.Hamiltonian(epg)
hep0.construct([[0.1, 1.5], [0.0, -2.7]])
ep.manual_H(hep0)
nup = np.where(epg.xyz[:, 1] > emg.center()[1])[0]  # + np.random.random(len(ZGNR))
ndn = np.where(epg.xyz[:, 1] < emg.center()[1])[0]  # + np.random.random(len(ZGNR))
MFH_EP = ep.Hubbard_electrode(nup, ndn, return_MFH=True, U=U, kT=kT)

#assert 1 == 0
Dev = SiP(dev.cell, dev.xyz, dev.atoms.Z, directory_name="Dev", elecs=[em, ep],
          Chem_Pot = [.0, .0], spin_pol='polarized', )
Dev.find_elec_inds()
rear = Dev._rearange_indices

HC = MFH_EM.tile(tx, axis=0)
HC.H.set_nsc([1, 1, 1])
MFH_HC = HubbardHamiltonian(
    HC.H.remove(remove).sub(rear),
    n=np.tile(MFH_EM.n, tx)[:, np.setdiff1d(np.arange(HC.H.na), remove)][:, rear],
    U=U,
    kT=kT,
)
negf = NEGF(MFH_HC, [(MFH_EM, "-A"), (MFH_EP, "+A")], Dev.elec_inds, )
dn = MFH_HC.converge(
    negf.calc_n_open,
    steps=10,
    mixer=sisl.mixing.PulayMixer(weight=0.1),
    tol=0.1,
)
dn = MFH_HC.converge(
    negf.calc_n_open,
    steps=100,
    mixer=sisl.mixing.PulayMixer(weight=1.0, history=7),
    tol=1e-8,
    print_info=True,
)
MFH_HC.H.shift(-MFH_HC.fermi_level())
Dev.manual_H(MFH_HC.H)
Dev.pickle('Dev')
np.save('spin-density.npy', MFH_HC.n)

# Contour = np.linspace(-8, 8, 201)+1j * 1e-2
# Dev.calculate_hubbard_transport(Contour, )




