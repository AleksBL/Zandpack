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
from Zandpack.FittingTools import find_correction

#assert 1 == 0
# In[]
Test = load(open('C1.Timedep','rb'))
#tbt = sisl.get_sile(Test.Device.dir + '/siesta.TBT.nc')

# In[]
Eg = np.linspace(-7,7,300)
#Test.reset_all_fits()
Emin,Emax = -5,5

NL = 11
E1 = np.linspace(Emin, Emax,NL)
E2 = np.linspace(Emin, Emax,NL)
init_E = [E1[None,:],E2[None,:]]
number = 0.2*(41/NL)**0.5
G = np.ones(init_E[0].shape)*number
G1 = G.copy()
G2 = G.copy()

G1[0,(NL//2-2):(NL//2+3)] = number/2


init_G = [G1,G2]
alpha_PO = 0.001

min_tol = np.zeros((1,NL))
min_tol[:,:] = 0.0#-0.12

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
          use_analytical_jac = False,          # Important for speed
          min_method = method,                 # Choose from any scipy.optimize.minimize method
          ebounds = (-7.5,7.5),                # bounds on centres
          wbounds = (0.0001, 3),               # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol = 1e-5,                          # any negative value with mean we fit all matrix elements of \\Gamma
          options = {'disp':True,              # Minimizer options
                     'maxiter':its, 
                     'gtol':1e-10, 
                     #'iprint':1,
                     },
          
          fit_real_part = False,               # 
          alpha_PO      = alpha_PO,            # Repulsion
          init_E        = init_E,              # Give initial ei's and gi's
          init_G        = init_G,
          which_e       = elecs
          )
    C = find_correction(Test)
    Test.Renormalise_H(C)

run_mini(0,'nelder-mead')
# In[]
def fixphase(v):
    nv = v.shape[0]
    for i in range(nv):
        sum_vi = v[:,i].sum()
        logsum = np.log(sum_vi).imag
        v[:,i]  *= np.exp(-1j * logsum)
        assert abs(v[:,i].sum().imag)< 1e-14

Test.tofile()



