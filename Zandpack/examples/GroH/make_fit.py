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
from lzma import open as Open
Test = load(Open("C1.xz", "rb"))
# In[]
#Eg = np.linspace(-7,7,300)
Emin,Emax = -8.5,8.5
NL        = 60

opts = {'height': 1.0, 
        'distance': 10
        }

E01, G01 = Test.PoleGuess(
        0,
        NL,
        -8.5,
         8.5,
        fact=0.8,
        cutoff=0.01,
        tol=0.1,
        decimals=3,
        pole_dist = 0.1,
        ik = 0,
        opts = opts,
)

E1     = np.vstack((E01,))
G1     = np.vstack((G01,))
E2     = np.linspace(Emin, Emax,NL)[None,:]
number = 0.2*(41/NL)**0.5
G      = np.ones((1,NL))*number
G2     = G.copy()

init_E = [E1, E2]
init_G = [G1, G2]

# In[]
alpha_PO = 0.001
min_tol = np.zeros((1,NL))
min_tol[:,:] = -10000.0
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
          ebounds = (-10.5,10.5),                # bounds on centres
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
    #C = find_correction(Test)
    #Test.Renormalise_H(C)
run_mini(0,'nelder-mead')
Test.curvefit_all(0.001)

Test.NO_fitted_lorentzians[0].iterative_PSD(maxit=250)
Test.NO_fitted_lorentzians[1].iterative_PSD()
Test.FitNO2O()
Test.tofile('GroH')


# In[]
# def fixphase(v):
#     nv = v.shape[0]
#     for i in range(nv):
#         sum_vi = v[:,i].sum()
#         logsum = np.log(sum_vi).imag
#         v[:,i]  *= np.exp(-1j * logsum)
#         assert abs(v[:,i].sum().imag)< 1e-14

# Test.tofile()


# Eg = np.linspace(-9,9,1000)
# ee = Test.bs2np(Test.NO_fitted_lorentzians[0].evaluate_Lorentzian_basis(Eg).eig(Test._Slices, hermitian=True)[0])
# eed = np.array([ee[:,:,i,i] for i in range(Test.n_orb)])
# plt.plot(Eg,np.log10(np.min(eed, axis=(1)).T)); plt.ylim(-3.2, 2)
# plt.xlabel(r'$E$ [eV]')
# plt.ylabel(r'$\log_{10}\lambda[\mathbf{\Gamma}_{gr}]$')
# plt.savefig(r'SpectrumGr.svg')
