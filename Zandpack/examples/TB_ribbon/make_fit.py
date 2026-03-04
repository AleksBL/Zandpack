#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:21:41 2022

@author: aleksander
"""


from pickle import load
import numpy as np
from Zandpack.FittingTools import rattle_lorentzians as rattle
import matplotlib.pyplot as plt
import sisl
import lzma

#assert 1 == 0


# In[]
R   = load(lzma.open('1Rib.xz','rb'))
#R.read_data()
tbt = sisl.get_sile(R.Device.dir + '/siesta.TBT.nc')

# In[]
Eg = np.linspace(-4,4,500)
#R.reset_all_fits()

NL = 61
Es = np.linspace(-4.0,4.0,NL)[None,:]
Gs = np.array([5.5/NL]*NL)[None,:] * (NL / 31 )**(-0.5)



init_E   = [Es,
            Es.copy()
           ]

init_G   = [Gs,
            Gs.copy()
            ]

alpha_PO = 0.01
min_tol = np.zeros((NL))
min_tol1, min_tol2 = min_tol.copy(),min_tol.copy()

def run_mini(its,method, which_e = None):
    R.Fit(fact = 0.6,                            # Redundant when we give init_E and init_G
          Fallback_W = 5.0,                    # Redundant\n",
          NumL = NL,                           # Not redundant\n",
          fit_mode = 'all',                    # Important to choose mode\n",
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [min_tol1,
                           min_tol2], # 
          use_analytical_jac = True,           # Important for speed
          min_method = method,                # Choose from any scipy.optimize.minimize method
          ebounds = (-4.5,4.5),                # bounds on centres
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
          which_e = which_e,
          )

# def revert(ie, ik):
#     Test.NO_fitted_lorentzians[ie].ei[ik]    = EI[ie][ik]
#     Test.NO_fitted_lorentzians[ie].gamma[ik] = WI[ie][ik]
run_mini(0,'trust-constr')
# for i in range(1):
#     run_mini(50,'trust-constr')
# for i in range(5):
#     run_mini(30,'SLSQP',        which_e=[0])
#     run_mini(30,'trust-constr', which_e=[0])

# run_mini(100,'SLSQP',       which_e=[0])



#R.diagonalise()
#R.get_propagation_quantities()
#R.get_dense_matrices_purenp(zero_tol = 1e-5)
#R.Check_input_to_ODE()
#f = R.make_f_purenp()
#R.write_to_file('1Rib_new')
