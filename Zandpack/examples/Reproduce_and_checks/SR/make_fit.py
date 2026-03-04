#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 08:59:22 2022

@author: aleksander
"""

from pickle import load
import numpy as np
import matplotlib.pyplot as plt
import sisl
from lzma import open

# In[]
Test = load(open('RSSE_raw.xz','rb'))
tbt = sisl.get_sile(Test.Device.dir + '/siesta.TBT.nc')


# In[]
Eg = np.linspace(-5,5,300)
NL = 51
E  = np.linspace(-5,5,NL)
number = 0.12
G1 = np.ones(NL)[None,:]*number
init_G = [G1,G1]
init_E = [E[None,:], E[None,:]]
alpha_PO = 0.001

min_tol = np.zeros((NL))
min_tol1, min_tol2 = min_tol.copy(),min_tol.copy()

# In[]
def run_mini(its,method, elecs = None):
    Test.Fit(fact = 0.6,                            # Redundant when we give init_E and init_G
          Fallback_W = 5.0,                    # Redundant\n",
          NumL = NL,                           # Not redundant\n",
          fit_mode = 'all',                    # Important to choose mode\n",
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [min_tol1,
                           min_tol2], # 
          use_analytical_jac = True,           # Important for speed
          min_method = method,                # Choose from any scipy.optimize.minimize method
          ebounds = (-7.5,7.5),                # bounds on centres
          wbounds = (0.0001, 3),                # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol = 1e-5,                         # any negative value with mean we fit all matrix elements of \\Gamma
          options = {'disp':True,              # Minimizer options
                     'maxiter':its, 
                     'gtol':1e-10, 
                     #'iprint':1,
                     }, 
          fit_real_part = False,              # 
          alpha_PO = alpha_PO,                # Repulsion
          init_E = init_E,                    # Give initial ei's and gi's
          init_G = init_G,
          which_e = elecs
          )


run_mini(0,'nelder-mead')
Test.tofile(name='STM2')
Test.pickle('Fit')

#Test.Check_input_to_ODE()

