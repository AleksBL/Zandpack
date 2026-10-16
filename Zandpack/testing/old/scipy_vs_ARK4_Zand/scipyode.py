#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:50:05 2022

@author: aleksander
"""

from TimedependentTransport.TimedependentTransport import scipy_ode
import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from shared import dH, delta




R = load(open('DevFit.Timedep','rb'))
R.diagonalise()
R.get_propagation_quantities()
R.get_dense_matrices(zero_tol = 1e-5)
R.Check_input_to_ODE()
f     = R.make_f_experimental()
sig   = R.sigma
psi   = R.Psi_vec
omega = R.omegas

t,dm,jl =  scipy_ode(f, sig, psi, omega, 0.0, np.linspace(0,20,50)+1e-7, dH, delta, R.Ixi, 
                      dH_given = True, method = 'RK45', dt_guess = None, 
                      atol = 1e-6, rtol = 1e-4)

plt.show()
plt.plot(t, jl)
plt.savefig('scipy_ode.png', dpi = 300)
np.savez('ScipyODE', t = t, dm = dm, jl = jl)
