#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:05:22 2022

@author: aleksander
"""

from make_device import D
import numpy as np
from siesta_python.siesta_python import SiP

[e.fdf() for e in D.elecs]
[e.run_siesta_electrode_in_dir() for e in D.elecs]
C = np.linspace(-3,3,300) + 1j *0.001
D.custom_tbtrans_contour = C
D.fdf()
D.run_siesta_in_dir()
D.run_tbtrans_in_dir()



Vi = np.linspace(-2,2,20)
Vi = Vi[np.argsort(np.abs(Vi))]

C = [D]
Vd= [0.0]
for v in Vi:
    v = np.round(v,5)
    Calc = SiP(D.lat,D.pos_real_space,D.s,
               directory_name=D.dir + '_' + str(v), 
               basis = D.basis, 
               NEGF_calc = True,
               Chem_Pot = [v/2, -v/2],
               elecs = D.elecs,
               solution_method = 'transiesta',
               kp=[1,1,1],
               print_mulliken=False,
               print_console=True,
               mesh_cutoff=200,
               kp_tbtrans = [1,1,1],
               trans_emin = -3.0, trans_emax = 3.0,
               trans_delta = 0.01
               )
    diff = v- np.array(Vd)
    idx = np.where(diff == diff.min())[0][0]
    Calc.copy_DM_from(C[idx])
    Calc.buffer_atoms = D.buffer_atoms
    Calc.elec_inds = D.elec_inds
    #Calc.Visualise()
    #Calc.custom_tbtrans_contour = D.custom_tbtrans_contour
    Calc.fdf()
    Calc.run_siesta_in_dir()
    Calc.run_tbtrans_in_dir()
    C += [Calc]
    Vd += [v]
