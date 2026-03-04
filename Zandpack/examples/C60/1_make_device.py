#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:47:07 2023

@author: aleksander
"""
import sisl
import matplotlib.pyplot as plt
from siesta_python.siesta_python import SiP
import numpy as np
from ase.build import molecule
mol = molecule('C60')
mol.write('Mol.xyz')
mol = sisl.get_sile('Mol.xyz').read_geometry()  # .rotate(90, [0,0,1])
mol = mol.move(-mol.center())
idx = np.argsort(mol.xyz[:,0])
mol = mol.sub(idx)
aAu = 2.4
tx = 14
txe = 5
basis = 'SZ'
offset = .25
AuChain = sisl.geom.sc(aAu, 'Au').add_vacuum(20, 1).add_vacuum(
    20, 2).add_vacuum(offset, 0).move([0.1, 0, 0])

em = AuChain.tile(txe, 0)
ep = AuChain.tile(txe, 0).move(AuChain.cell[0] * (tx-txe)).move([offset, 0, 0])

Base = AuChain.tile(tx, 0).sub(
    [i for i in range(txe)] + [tx-1-i for i in range(txe)])
Base = Base.move([offset, 0, 0], [i+txe for i in range(txe)])
mol = mol.move(Base.center() - np.array([0.0, 0.0, 0.0]))

Base = Base.add(mol)
T = Base.cell.sum(axis=0)*np.array([0, .5, .5])
Base = Base.move(T)
em = em.move(T)
ep = ep.move(T)

slako = '../pp'
EM = SiP(em.cell, em.xyz, em.atoms.Z,
         directory_name='EM', sl='EM',
         kp=[30, 1, 1], semi_inf='-a1',
         dm_tol='1e-5', basis=basis,
         pp_path=slako,mpi='mpirun '
         )

EP = SiP(ep.cell, ep.xyz, ep.atoms.Z,
         directory_name='EP', sl='EP',
         kp=[30, 1, 1], semi_inf='+a1',
         dm_tol='1e-5', basis=basis,
         pp_path=slako,mpi='mpirun '
         )

Dev = SiP(Base.cell, Base.xyz, Base.atoms.Z,
          directory_name='Device',
          elecs=[EM, EP], solution_method='transiesta',
          dm_tol='1e-5',
          pp_path=slako, kp=[1, 1, 1],
          Chem_Pot=[0.0, 0.0],
          trans_emin=-3, basis=basis,
          trans_emax=3,
          trans_delta=0.025,
          save_SE=True,
          mpi='mpirun '
          )

Dev.find_elec_inds()
EM.fdf()
EM.run_siesta_in_dir()
EP.fdf()
EP.run_siesta_in_dir()
hem, _ = EM.to_sisl('fromDFT')
hep, _ = EP.to_sisl('fromDFT')
hem.set_nsc((3, 1, 1))
hep.set_nsc((3, 1, 1))
EM.manual_H(hem)
EP.manual_H(hep)
Dev.copy_DM_from("savedDM")
Dev.fdf()
Dev.ase_visualise()
Dev.run_siesta_in_dir()
Dev.pickle('Dev')
