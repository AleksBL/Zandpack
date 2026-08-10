#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:02:23 2022

@author: aleksander
"""

from TimedependentTransport.Optimized_RK45 import AdaptiveRK4_Opti as RK4
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from TimedependentTransport.Pulses import zero_dH, zero_bias




# R = load(open('1Rib_fitted.Timedep','rb'))

# R.diagonalise()
# R.get_propagation_quantities()
# R.get_dense_matrices_purenp(zero_tol = 1e-5)
# R.Check_input_to_ODE()
# f = R.make_f_purenp()
# #R.write_to_file('1Rib_file')

# sig = R.sigma
# psi = R.Psi_vec
# omg = R.omegas

# def bias(t,a):
#     return 

# _t0, _d0 = RK4(f, sig, psi, omg, 1e-5, 0, 10, zero_dH, zero_bias, R.Ixi, name = 'GB')
# plt.plot(_t0, _d0['current_left'])
# plt.show()
