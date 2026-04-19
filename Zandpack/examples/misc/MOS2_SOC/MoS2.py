#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:59:31 2024

@author: investigator
"""

import sisl
import numpy as np
import matplotlib.pyplot as plt
from siesta_python.siesta_python import SiP
import sisl
g = sisl.get_sile('MoS2.xyz').read_geometry()
clc1 = SiP(g.cell, g.xyz, g.atoms.Z, 
          directory_name='NOSOC', kp = [12,12,1], 
          mpi='mpirun', basis='DZP')
clc2 = SiP(g.cell, g.xyz, g.atoms.Z, 
           directory_name='SOC', kp = [12,12,1], 
           mpi='mpirun', basis='DZP', spin_pol='spin-orbit')
clc2.write_more_fdf(['Spin.OrbitStrength 1.0'])
clc1.fdf(); clc1.run_siesta_in_dir()
clc2.fdf(); clc2.run_siesta_in_dir()
H1 = clc1.read_TSHS(); H2 = clc2.read_TSHS()
# In[]
nk   = 200
band = sisl.BandStructure(H1, [[0,0,0],   [1/3,1/3,0], 
                               [0.5,0,0], [0,  0,  0]], 
                          nk, ['Gamma', 'K', 'M','Gamma'])
eig1 = np.zeros((nk, H1.no  ))
eig2 = np.zeros((nk, H2.no*2))
for i in range(nk):
    eig1[i] = H1.eigh(k = band.k[i])
    eig2[i] = H2.eigh(k = band.k[i])
plt.plot(eig1,color='red')
plt.plot(eig2,color='blue') 
plt.ylim([-2,2])