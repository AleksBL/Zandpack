#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:58:14 2023

@author: aleksander
"""

from pickle import load
import lzma
import numpy as np
from Zandpack.FittingTools import find_correction
from Zandpack.plot import plt
#assert 1 == 0
# In[]
Ng = 2
with lzma.open('T2T_G'+str(Ng)+'.xz','rb') as f:
    R = load(f)
# In[]
Eg   = np.linspace(-8.5,8.5,250)
NL   = 25
opts = {'height':1.0, 
        'distance':5, }
Emin, Emax = -7.0, 5.0
pdist   = 0.1
fact    = 1.3
pfact   = 0.9
cf      = 0.1
fm      = 'linear'
E1, G1  = R.PoleGuess(0, NL, Emin, Emax,fact=fact, cutoff=cf, tol = 1.0, decimals=3, pole_dist = pdist, pole_fact = pfact, opts = opts)
E2, G2  = R.PoleGuess(1, NL, Emin, Emax,fact=fact, cutoff=cf, tol = 1.0, decimals=3, pole_dist = pdist, pole_fact = pfact, opts = opts)
#E1 = np.linspace(Emin, Emax, NL); E2 = E1.copy()
#G1 = np.ones( NL)*0.4 ; G2 = G1.copy()

init_E = [E1[None,:], E2[None,:]]
init_G = [G1[None,:], G2[None,:]]
alpha_PO = 0.0
min_tol  = np.zeros(NL)[None,:]
min_tol[:,:] = -1000.0
min_tol1, min_tol2 = min_tol.copy(),min_tol.copy()
# In[]
sb1     = {}
sb2     = {}
tol_ele = 1e-3
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
          ebounds    = (-10.,10.),                # bounds on centres
          wbounds    = (0.001, 2),                # bounds on widths
          gbounds = (None, None),              # bounds on sizes, redundant right now
          tol     = tol_ele,                      # any negative value with mean we fit all matrix elements of \\Gamma
          options = {'disp':3,              # Minimizer options
                     'maxiter':its, 
                     'gtol':1e-5,
                     #'ftol':1e-3,
                     'verbose':3,
                     },
          fit_real_part = False,               # 
          alpha_PO = alpha_PO,                 # Repulsion
          init_E   = init_E,                   # Give initial ei's and gi's
          init_G   = init_G,
          which_e  = which_e,
          )

run_mini(0,'trust-constr')
R.curvefit_all(.0005)
R.Inspect_transmission_from_hilbert_transform(E=Eg,eta=1.5e-2)
plt.title('Before')
plt.show()
R.NO_fitted_lorentzians[0].iterative_PSD(n=30,nW=15, eps=0.05, 
                                         maxit = 40)
R.NO_fitted_lorentzians[1].iterative_PSD(n=30,nW=15, eps=0.05, 
                                         maxit = 40)
R.FitNO2O()
C = find_correction(R, Emin = -1.0, Emax = 1.0)
R.Renormalise_H(C)
R.Inspect_transmission_from_hilbert_transform(E=Eg,eta=1.5e-2)
plt.title('After')
plt.show()
R.tofile('TDT2T_G'+str(Ng))