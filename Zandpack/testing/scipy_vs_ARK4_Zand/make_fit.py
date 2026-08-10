#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 22:17:52 2022

@author: aleksander
"""

from pickle import load
import numpy as np
from TimedependentTransport.FittingTools import rattle_lorentzians as rattle
import matplotlib.pyplot as plt
import sisl
from TimedependentTransport.FittingTools import f_space, find_correction



# In[]
R   = load(open('Dev1.Timedep','rb'))
#R.read_data()
tbt = sisl.get_sile(R.Device.dir + '/siesta.TBT.nc')

# In[]
Eg = np.linspace(-4,4,100)
#Test.reset_all_fits()

NL       =  11

Es = np.linspace(-3.5,3.5,NL)[None,:][[0,0,0]]
Gs = np.array([3.0/NL]*NL)[None,:][[0,0,0]]

init_E   = [Es, #f_space(dist, -4, 4, 0.45)[None,:],
            Es.copy(),#np.linspace(-4.0,4.0,NL)[None,:],
            Es.copy()#np.linspace(-4.0,4.0,NL)[None,:],
            
            #f_space(dist, -4, 4, 0.45)[None,:]
            #np.linspace(-4.0,4.0,NL)[None,:],
           ]

init_G   = [Gs,#0.2/dist(init_E[0]),
            Gs.copy(),#np.array([3.0/NL]*NL)[None,:],
            Gs.copy()#np.array([3.0/NL]*NL)[None,:],
            #0.2/dist(init_E[1]) 
            #np.array([3.0/NL]*NL)[None,:],
            ]




# NL = len(init_E[0][0])
# assert 1 == 0

alpha_PO = 0.01

min_tol = np.zeros((3,NL))
#min_tol[:] = -1
min_tol1, min_tol2, min_tol3 = min_tol.copy(),min_tol.copy(), min_tol.copy()


def run_mini(its,method, which_e = None):
    R.Fit(fact = 0.6,                            # Redundant when we give init_E and init_G
          Fallback_W = 5.0,                    # Redundant\n",
          NumL = NL,                           # Not redundant\n",
          fit_mode = 'all',                    # Important to choose mode\n",
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [min_tol1,
                           min_tol2,
                           min_tol3
                           ], # 
          use_analytical_jac = True,           # Important for speed
          min_method = method,                # Choose from any scipy.optimize.minimize method
          ebounds = (-3.9,3,9),                # bounds on centres
          wbounds = (0.00001, 1),                # bounds on widths
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
          which_e = which_e,
          )
    
# def revert(ie, ik):
#     Test.NO_fitted_lorentzians[ie].ei[ik]    = EI[ie][ik]
#     Test.NO_fitted_lorentzians[ie].gamma[ik] = WI[ie][ik]
run_mini(0,'trust-constr')
C = find_correction(R)
R.Renormalise_H(C)
R.pickle('DevFit')

