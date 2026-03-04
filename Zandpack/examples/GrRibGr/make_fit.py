#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:27:27 2023

@author: aleksander
"""


from pickle import load
import lzma
import numpy as np
from Zandpack.FittingTools import find_correction
from Zandpack.plot import plt

# In[]
with lzma.open('GrRibGr2.xz','rb') as f:
    R = load(f)
# In[]
Eg = np.linspace(-6.5,6.5,250)
#R.reset_all_fits()
NL     = 61
##E1    = np.linspace(-6.8, 6.8,NL)[None,:]
opts = {'height':5.0, 'distance':3}

E1,G1  = R.PoleGuess(0, NL, -7.8, 7.8,fact=0.8, cutoff=0.1, tol = 1.0, decimals=3, pole_dist = 0.1, pole_fact = 1.0, opts = opts)
E2,G2  = R.PoleGuess(1, NL, -7.8, 7.8,fact=0.8, cutoff=0.1, tol = 1.0, decimals=3, pole_dist = 0.1, pole_fact = 1.0, opts = opts)
init_E = [E1[None,:], E2[None,:]]
init_G = [G1[None,:], G2[None,:]]

alpha_PO = 0.0
min_tol  = np.zeros(NL)
min_tol1, min_tol2 = min_tol.copy(),min_tol.copy()

# In[]
sb1       = {}
sb2       = {}
tol_ele   = 1e-1

# for i in range(NL):
#     W = G1[i]
#     if W<0.11:
#         sb1.update({(0, i):((E1[i]-2*W, E1[i]+2*W),(W*0.5, W*2.0))})
#     else:
#         sb1.update({(0, i):((E1[i], E1[i]),(W, W))})

# for i in range(NL):
#     W = G2[i]
#     if W<0.11:
#         sb2.update({(0, i):((E2[i]-2*W, E2[i]+2*W),(W*0.5, W*2.0))})
#     else:
#         sb2.update({(0, i):((E2[i], E2[i]),(W, W ))})

def run_mini(its,method, which_e = None):
    R.Fit(fact = 1.0,                          # Redundant when we give init_E and init_G
          Fallback_W = 10.0,                    # Redundant\n",
          NumL = NL,                           # Not redundant\n",
          fit_mode      = 'all',               # Important to choose mode\n",
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [min_tol1,
                           min_tol2], # 
          use_analytical_jac = True,           # Important for speed
          min_method = method,                 # Choose from any scipy.optimize.minimize method
          ebounds    = (-7.5,7.5),                # bounds on centres
          wbounds    = (0.001, 5),                # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol     = tol_ele,                      # any negative value with mean we fit all matrix elements of \\Gamma
          options = {'disp':3,              # Minimizer options
                     'maxiter':its, 
                     'gtol':1e-5,
                     'ftol':1e-3,
              #       'iprint':1,
                     'verbose':3,
                     #'optimparallel':True
                     },
          specific_bounds = [sb1, sb2],
          fit_real_part = False,               # 
          alpha_PO = alpha_PO,                 # Repulsion
          init_E   = init_E,                   # Give initial ei's and gi's
          init_G   = init_G,
          which_e  = which_e,
          #exc_idx = [exc_idx1, exc_idx2]
          )
    #C = find_correction(R, Emin = -1, Emax = 1)
    #R.Renormalise_H(C)

run_mini(0,'COBYLA')
#R.Inspect_transmission_from_hilbert_transform(E=Eg[::3],eta=5e-3)
R.curvefit_all(0.1)
R.NO_fitted_lorentzians[0].iterative_PSD(maxit=101);
R.NO_fitted_lorentzians[1].iterative_PSD(maxit=101);
C = find_correction(R, Emin = -1.0, Emax = 1.0)
R.Renormalise_H(C)

# R.tofile('GrRibGr')
# R.pickle('Testing')