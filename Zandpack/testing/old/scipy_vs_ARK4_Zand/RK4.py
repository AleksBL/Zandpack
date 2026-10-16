#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:42:24 2022

@author: aleksander
"""

from TimedependentTransport.Optimized_RK45 import AdaptiveRK4_Opti as RK4
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from shared import dH, delta

R = load(open('DevFit.Timedep','rb'))

R.diagonalise()
R.get_propagation_quantities()
R.get_dense_matrices_purenp(zero_tol = 1e-5)
R.Check_input_to_ODE()
f = R.make_f_purenp()

sig = R.sigma
psi = R.Psi_vec
omg = R.omegas

_t0, _d0 = RK4(f, sig, psi, omg, 1e-5, 0, 20, dH, delta, R.Ixi, 
               name = 'GB', elec_names = ['L','R', 'O'],
               )
               
plt.plot(_t0, _d0['current_L'])
# plt.plot(_t0, _d0['current_R'])
# plt.plot(_t0, _d0['current_O'])

plt.show()
