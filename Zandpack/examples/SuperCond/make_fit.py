#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:33:19 2022

@author: aleksander
"""


from pickle import load
import numpy as np
from Zandpack.FittingTools import rattle_lorentzians as rattle
import matplotlib.pyplot as plt
import sisl
import lzma


# In[]
with lzma.open('Test.xz','rb') as f:
    Test = load(f)
tbt = sisl.get_sile(Test.Device.dir + '/siesta.TBT.nc')


# In[]
Eg = np.linspace(-8,8,300)
#Test.reset_all_fits()

NL = 21
init_E = [np.linspace(-8.5,8.5,NL)[None,:]]*2
init_G = [np.array([3.0*3.5/NL]*NL)[None,:]]*2
alpha_PO = 0.0001

min_tol = np.zeros((NL))
#min_tol[:] = -1
min_tol1, min_tol2 = min_tol.copy(),min_tol.copy()

def run_mini(its,method):
    Test.Fit(fact = 0.6,                            # Redundant when we give init_E and init_G
          Fallback_W = 5.0,                    # Redundant\n",
          NumL = NL,                           # Not redundant\n",
          fit_mode = 'all',                    # Important to choose mode\n",
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [min_tol1,
                           min_tol2], # 
          use_analytical_jac = True,           # Important for speed
          min_method = method,                # Choose from any scipy.optimize.minimize method
          ebounds = (-6.5,6.5),                # bounds on centres
          wbounds = (0.0001, 3),                # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol = -1e-2,                         # any negative value with mean we fit all matrix elements of \\Gamma
          options = {'disp':True,              # Minimizer options
                     'maxiter':its, 
                     'gtol':1e-10, 
                     #'iprint':1,
                     }, 
          fit_real_part = False,              # 
          alpha_PO = alpha_PO,                # Repulsion
          init_E = init_E,                    # Give initial ei's and gi's
          init_G = init_G,
          )

run_mini(0,'trust-constr')

Test.tofile('2Chain')