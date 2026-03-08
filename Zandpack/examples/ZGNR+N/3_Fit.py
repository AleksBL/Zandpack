#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:18:37 2023

@author: aleksander
"""

from pickle import load
import lzma
import numpy as np
from Zandpack.FittingTools import rattle_lorentzians as rattle
import matplotlib.pyplot as plt
import sisl
from Zandpack.FittingTools import find_correction
ensure_psd = True
R    = load(lzma.open('AGNR.xz','rb'))
# assert 1 == 0
R.adapt_sampling()
Eg   = np.linspace(-6.9,6.9,250)
# R.reset_all_fits()
Name = 'AGNRTD'
NL, Emin, Emax = 30, -6.5, 6.5
opts, fm= {'height':1.0, 'distance':5}, 'linear'
pdist, fact, pfact, cf = 0.05, 1.2, 1.0, 0.01
E1,G1=R.PoleGuess(0,NL,Emin,Emax,fact=fact,cutoff=cf,
                  tol=.01,decimals=3,pole_dist=pdist,
                  pole_fact=pfact,opts=opts)
E2,G2=R.PoleGuess(1,NL,Emin,Emax,fact=fact,cutoff=cf,
                  tol=.01,decimals=3,pole_dist=pdist,
                  pole_fact=pfact,opts=opts)

E1,E2 = E1[None,:], E2[None,:]
G1,G2 = G1[None,:], G2[None,:]
# E1   = np.linspace(-5.8,5.8,NL)[None,:]
# E2   = np.linspace(-5.8,5.8,NL)[None,:]
# E3   = np.linspace(-5.8,5.8,NL)[None,:]
# E1[0,20] -= 0.085
# E2[0,20] -= 0.085
init_E   = [E1, E2]
init_G   = [G1, G2]
# G1       = np.array([5.5/NL]*NL)[None,:]
# G2       = np.array([5.5/NL]*NL)[None,:]
# G3       = np.array([5.5/NL]*NL)[None,:]
# G1[0,20] = .5/NL
# G2[0,20] = .5/NL
alpha_PO = 0.0001
min_tol  = np.zeros(NL); min_tol[:]=-1000.0
min_tol1, min_tol2 = min_tol.copy(),min_tol.copy()

sb1,sb2,sb3 = {},{},{}
eps         = 3e-1
pc1,pc2     = 0.75, 1.5
tol_ele     = 1e-10
for i in range(NL):
   sb1.update({(0, i):((E1[0,i]-eps, E1[0,i]+eps),
                       (G1[0,i]*pc1, G1[0,i]*pc2))})
   sb2.update({(0, i):((E2[0,i]-eps, E2[0,i]+eps),
                       (G2[0,i]*pc1, G2[0,i]*pc2))})

# sb1.update({(0,20):((E1[0,i]-eps, E1[0,i]+eps),(1.0/NL-eps, 1.0/NL+eps))})
#for i in range(NL):
#    sb2.update({(0, i):((E1[0,i]-eps, E1[0,i]+eps),(0.01,0.5))})
# In[]
def run_mini(its,method, which_e = None):
    R.Fit(fact = 0.6,                            # Redundant when we give init_E and init_G
          Fallback_W = 5.0,                    # Redundant\n",
          NumL = NL,                           # Not redundant\n",
          fit_mode      = 'all',               # Important to choose mode\n",
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [min_tol1,
                           min_tol2], # 
          use_analytical_jac = False,           # Important for speed
          min_method = method,                 # Choose from any scipy.optimize.minimize method
          ebounds = (-7.5,7.5),                # bounds on centres
          wbounds = (0.01, 1),                 # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol     = tol_ele,                      # any negative value with mean we fit all matrix elements of \\Gamma
          options = {'disp':True,              # Minimizer options
                     'maxiter':its, 
                     'gtol':1e-10, 
          #           'iprint':1,
                     },
          specific_bounds = [sb1,sb2,sb3],
          fit_real_part = False,               # 
          alpha_PO = alpha_PO,                 # Repulsion
          init_E   = init_E,                   # Give initial ei's and gi's
          init_G   = init_G,
          which_e  = which_e
          )

run_mini(0,'SLSQP')
R.curvefit_all(0.0001)

if ensure_psd:
    R.NO_fitted_lorentzians[0].iterative_PSD(
        maxit=61, n=40,nW=15,eps=0.05,
        lbtol = -0.001,fact=.5,add_last = False)
    R.NO_fitted_lorentzians[1].iterative_PSD(
        maxit=61, n=40,nW=15,
        lbtol = -0.001,fact=.5,eps=0.05,
        add_last = False)

R.FitNO2O()
C = find_correction(R, Emin = -2.0, Emax=2.0)
R.Renormalise_H(C)
R.Inspect_transmission_from_hilbert_transform(E=Eg,
                                              eta=1e-2, 
                                              figure = 'T.svg')

    
R.tofile('ZGNR_N')