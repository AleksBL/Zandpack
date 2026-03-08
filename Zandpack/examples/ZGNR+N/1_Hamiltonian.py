#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:57:13 2023

@author: aleksander
"""

import sisl
import numpy as np
import matplotlib.pyplot as plt
from siesta_python.siesta_python import SiP
from siesta_python.funcs import zigzag_g
from Structures.Structures import AlTip_EP, AlTip_Dev
from ase.build import molecule
from scipy.linalg import eigh, fractional_matrix_power

Hatom = molecule('H')

CT = AlTip_Dev.center()
CT[2]=0
AlTip_Dev = AlTip_Dev.move(-CT)
AlTip_EP  = AlTip_EP.move(-CT).add_vacuum(10,1)
CT = AlTip_Dev.center()

tip_z    = 3.0

AlTip_Dev = AlTip_Dev.sub([i for i in range(AlTip_Dev.na) if AlTip_Dev.xyz[i,2]>CT[2]])
_z        = AlTip_Dev.xyz[:,2].min()
g_al = AlTip_Dev.move([0,0,-_z]).move([0,0,tip_z])
geo  = AlTip_EP.move( [0,0,-_z]).move([0,0,tip_z])
#assert 1 == 0

move_tip= np.array([0,0,0])
txd     = 6

angmom  = {'C':'p',
           'H':'s',
           'N':'p'
           # 'Al':'p'
           }

ThirdOrderFull = 'No'# 'Yes'
HubDer  = None
DampCor = None
slako = '/home/aleks/Desktop/Zandpack_Videos/dftb+/matsci-0-3/'# /home/aleksander/Desktop/slako/3ob/3ob-3-1/'

g    = sisl.geom.nanoribbon(width=4, bond=1.42, atoms = 6, kind='zigzag').add_vacuum(10,1).add_vacuum(50,2)
g.xyz[:, :]+= 1e-3

_t0  = 3
gem  = g.tile(_t0,0)
gep  = g.tile(_t0,0).move(_t0 * g.cell[0]*(txd-1))
gd   = g.tile(_t0 * txd, 0)
CR   = gd.center()

# In[]
EM  = SiP(gem.cell, gem.xyz, gem.atoms.Z, 
          directory_name = 'EM',
          sl = 'EM', dm_tol   ='1e-5',
          pp_path  = slako,
          semi_inf = '-a1',
          kp = [25,1,1] )
EP  = SiP(gep.cell, gep.xyz, gep.atoms.Z, 
          directory_name = 'EP',
          sl = 'EP', dm_tol   ='1e-5',
          pp_path  = slako,
          semi_inf = '+a1',
          kp = [25,1,1] )

# In[]
def cond(r):
    print(r)
    if r[2]<10.5:
        return True
    return False


EM.Passivate(1, 1.05, condition=cond)
EP.Passivate(1, 1.05, condition=cond)
EM.write_and_run_dftb_in_dir(angmom, ThirdOrderFull=ThirdOrderFull, HubDeriv=HubDer, DampCor=DampCor)
EP.write_and_run_dftb_in_dir(angmom, ThirdOrderFull=ThirdOrderFull, HubDeriv=HubDer, DampCor=DampCor)

hem  = EM.dftb2sisl()
hep  = EP.dftb2sisl()
bzm  = sisl.MonkhorstPack(hem, [50,1,1])
bzp  = sisl.MonkhorstPack(hep, [50,1,1])

mum  = hem.fermi_level(bz = bzm, q = EM.ideal_cell_charge( {'C':4.0, 'H':1.0} ))
mup  = hep.fermi_level(bz = bzp, q = EP.ideal_cell_charge( {'C':4.0, 'H':1.0} ))

hem.shift(-mum-0.01)
hep.shift(-mup-0.01)
hem.set_nsc((3,1,1))
hep.set_nsc((3,1,1))

EM.manual_H(hem)
EP.manual_H(hep)

bandm = sisl.BandStructure(hem, [[.0, .0, .0], [.5, .0, .0]],100)
bandp = sisl.BandStructure(hep, [[.0, .0, .0], [.5, .0, .0]],100)
em,ep= np.zeros((100,hem.no)), np.zeros((100, hep.no))# , np.zeros((100, heo.no))
for i in range(100):
    em[i] = hem.eigh(k=bandm.k[i])
    ep[i] = hep.eigh(k=bandp.k[i])



Dev = SiP(gd.cell, gd.xyz, gd.atoms.Z, 
          directory_name = 'Dev',dm_tol='1e-5',
          pp_path  = slako,
          solution_method='transiesta',
          elecs    = [EM, EP],
          Chem_Pot    = [.0, .0, .0],
          trans_emin  = -4.0,
          trans_emax  =  4.0,
          trans_delta =  0.05,
          kp = [1,1,1],
          kp_tbtrans  = [1,1,1],
          save_SE=True
          )

Dev.Passivate(1, 1.05, condition = cond)
CDev = Dev.to_sisl().center()
dist = np.linalg.norm(CDev - Dev.pos_real_space, axis=1)
i = np.where(dist == dist.min())[0][0]
Dev.s[i] = 7
# assert 1 == 0

Dev.find_elec_inds()
Dev.write_and_run_dftb_in_dir(angmom, ThirdOrderFull=ThirdOrderFull, HubDeriv=HubDer, DampCor=DampCor)
Hd  = Dev.dftb2sisl(tol = 1e-10)
hk  = Dev.fast_dftb_hk()
bzd = sisl.MonkhorstPack(Hd, [10,1,1])
mud = Hd.fermi_level(bz = bzd, 
                     q  = Dev.ideal_cell_charge({'N':5.0,'C':4.0, 'H':1.0})
                     )

Hd.shift(-mud)
dm_d = np.zeros((Hd.no, Hd.no),dtype=complex)
Kv = np.linspace(0,0.5,2)[0:1]
nelecs = 0.0

for _k in Kv:
    k   = (_k,0,0)
    HK  = Hd.Hk(k=k).toarray()
    SK  = Hd.Sk(k=k).toarray()
    e,v = eigh(HK,SK)
    FD  =  np.diag(1/(1+np.exp(e/0.025)))
    dm_d_k = v@FD@(v.conj().T)
    nelecs += (dm_d_k*SK).sum()/1
    dm_d += dm_d_k/1

np.savez_compressed('StSt',
                     mum  = mum,
                     mup  = mup,
                     # muo  = muo,
                     mud  = mud,
                     DM_d = dm_d, 
                     SK   = SK)

Hd.set_nsc((1,1,1))
Dev.manual_H(Hd)
Dev.pickle('Dev')

# Dev.fdf(eta=1e-4)
# Dev.run_siesta_in_dir()
# Dev.run_tbt_analyze_in_dir()
# Dev.run_tbtrans_in_dir(DOS_GF=True)
# tbt = Dev.read_tbt()
# E = tbt.E + 1j*1e-4#np.linspace(-4,4, 500)+1j*1e-3

#hd    = Hd.Hk().toarray()
#sd    = Hd.Sk().toarray()
#SE1   = sisl.RecursiveSI(hem, '-A')
#SE2   = sisl.RecursiveSI(hep, '+A')

# inds1 = []
# for i in Dev.elec_inds[0]:
#     inds1 +=[j for j in range(Hd.a2o(i),Hd.a2o(i+1))]
# inds2 = []
# for i in Dev.elec_inds[1]:
#     inds2 +=[j for j in range(Hd.a2o(i),Hd.a2o(i+1))]

# Dev.from_custom(hd[None,:,:], [SE1,SE2], 
#                 E, 
#                 np.array([[0.,0.,0.]]), 
#                 S = sd[None,:,:], 
#                 SE_inds = [inds1,inds2]
#                 )

# tbt = Dev.read_tbt()
# f   = np.load(Dev.dir+'/siesta.fakeTBT.npz')
# E,T =f['E'], f['transmission']
# plt.plot(tbt.E, tbt.transmission())
# plt.plot(E,T)

# In[]
# hemp = hem#hem.sub_orbital([i for i in range(hem.na) if hem.atoms.Z[i] == 6], [2])
# hepp = hep#hep.sub_orbital([i for i in range(hep.na) if hep.atoms.Z[i] == 6], [2])
# #Hdp  =  Hd.sub_orbital([i for i in range(Hd.na)  if Hd.atoms.Z[i] == 6], [2])

# Em1 = SiP(hemp.cell, hemp.xyz, hemp.atoms.Z,
#           directory_name = 'EMp', 
#           sl = 'EMp',
#           semi_inf = '-a1')

# Ep1 = SiP(hepp.cell, hepp.xyz, hepp.atoms.Z,
#           directory_name = 'EPp', 
#           sl = 'EPp',
#           semi_inf = '+a1')

# Em1.manual_H(hemp)
# Ep1.manual_H(hepp)

# Hdp = hemp.tile(txd,0)
# plt.show()
# band = sisl.BandStructure(hemp, [[0,0,0],[0.5,0,0.0]],400)
# e    = np.zeros((400, hemp.no))
# for i,k in enumerate(band):
#     e[i] = hemp.eigh(k = k )
# plt.plot(e)
# plt.ylim(-3, 3)

# plt.title('p-part')

# Devp = SiP(Hdp.cell,Hdp.xyz, Hdp.atoms.Z,
#             directory_name='Devp',
#             elecs = [Em1, Ep1],
#             Chem_Pot = [.0, .0],
#             trans_emin = -4.0,
#             trans_emax = 4.0,
#             trans_delta= 0.05
#             )

# Devp.find_elec_inds()
# Hdp = Hdp.sub(Devp._rearange_indices)

# Devp.manual_H(Hdp)
# Devp.fdf(eta=1e-3)
# Devp.run_tbt_analyze_in_dir()
# Devp.run_tbtrans_in_dir()











