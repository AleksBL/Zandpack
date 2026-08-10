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

# In[]
R   = load(lzma.open('1Rib.xz','rb'))
tbt = sisl.get_sile(R.Device.dir + '/siesta.TBT.nc')

# In[]
Eg = np.linspace(-8,8,500)
NL = 61
opts= {'height':1.0, 'distance':5}
Emin, Emax = -8.0, 8.
pdist, fact, pfact, cf = 0.1,0.9,0.8,0.1
fm      = 'linear'
E1,G1=R.PoleGuess(0,NL,Emin,Emax,fact=fact,cutoff=cf,
                  tol=1.0,decimals=3,pole_dist=pdist,
                  pole_fact=pfact,opts=opts)
E2,G2=R.PoleGuess(1,NL,Emin,Emax,fact=fact,cutoff=cf,
                  tol=1.0,decimals=3,pole_dist=pdist,
                  pole_fact=pfact,opts=opts)
init_E = [E1[None,:], E2[None,:]]
init_G = [G1[None,:], G2[None,:]]
alpha_PO = 0.01
min_tol  = np.zeros((1,NL))
min_tol[:,:] = -0.0
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
          ebounds = (-9.5,9.5),                # bounds on centres
          wbounds = (0.0001, 3),                # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol = 1e-3,                         # any negative value with mean we fit all matrix elements of \\Gamma
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

run_mini(0,'trust-constr')

R.pickle('Fit_5')
R.tofile('Rib_5')
