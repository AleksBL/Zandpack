#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:29:18 2024

@author: investigator
"""

import numpy as np
import matplotlib.pyplot as plt
from siesta_python.siesta_python import SiP
import sisl
AuTip_Dev = sisl.get_sile("Dev.xyz").read_geometry()
AuTip_EM = sisl.get_sile("EM.xyz").read_geometry()
AuTip_EP = sisl.get_sile("EP.xyz").read_geometry()


Ng = 0
dz = 2.0
T   = np.array([5,8,0])
gem = AuTip_EM.move(T)
gep = AuTip_EP.move(T)
gd  = AuTip_Dev.move(T)
basis = 'SZ'

C   = gd.center()
gd  = gd.move([0,0,Ng*dz], np.where(gd.xyz[:,2]>C[2])[0])
gep = gep.move([0,0,Ng*dz])
cell, xyz, z = gd.cell.copy(), gd.xyz.copy(), gd.atoms.Z.copy()
cell[2,2] += Ng*dz
newC = C + np.array([0,0,Ng * dz/2])
if   Ng == 0:
    pass
elif Ng == 1:
    
    xyz = np.vstack([xyz, newC[None,:]])
    z   = np.hstack([z, -79])
else:
    zg  = np.linspace(newC[2]-Ng*dz/2+1,newC[2] + Ng*dz/2-1, Ng)
    newz= np.zeros((Ng, 3))
    newz[:,0:2] = newC[0:2]
    newz[:,2]   = zg
    xyz = np.vstack([xyz,newz ])
    z   = np.hstack([z, [-79]*Ng])

EM  = SiP(gem.cell, gem.xyz, gem.atoms.Z, 
          directory_name = 'EM',
          sl = 'EM',basis=basis,pp_path = '../pp_psf',
          semi_inf = '-a3',
          kp = [1,1,50], )

EP  = SiP(gep.cell, gep.xyz, gep.atoms.Z, 
          directory_name = 'EP',
          sl = 'EP',basis=basis,pp_path = '../pp_psf',
          semi_inf = '+a3',
          kp = [1,1,50], )

Dev = SiP(cell, xyz, z, 
          directory_name = 'Dev'+str(Ng),
          solution_method='transiesta',
          elecs       = [EM, EP],basis=basis,
          Chem_Pot    = [.0, .0],
          kp          = [1,1,1],
          kp_tbtrans  = [1,1,1],
          save_SE     = True,pp_path = '../pp_psf', )

Dev.find_elec_inds()
EM.fdf()
EP.fdf()
Dev.fdf()
EM.run_siesta_in_dir()
EP.run_siesta_in_dir()
Dev.run_siesta_in_dir()
H = Dev.read_TSHS()
H.set_nsc((1,1,1))
H.write(Dev.dir+'/siesta.TSHS')
Dev.pickle('Dev' + str(Ng))
