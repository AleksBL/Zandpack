#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:18:37 2023

@author: aleksander
"""

from pickle import load
import numpy as np
from Zandpack.FittingTools import rattle_lorentzians as rattle
import matplotlib.pyplot as plt
import sisl
from Zandpack.FittingTools import find_correction
from lzma import open as Open

R = load(Open("Pohl.xz", "rb"))
Eg = np.linspace(-5.9, 5.9, 250)
# R.reset_all_fits()
Name = "AGNRTD"
NL   = 9
E1, G1 = R.PoleGuess(
    0,
    NL,
    -5.8,
    5.8,
    fact=1.0,
    pole_dist=0.1,
    cutoff=0.1,
    tol=1.0,
    decimals=3,
)
E2, G2 = R.PoleGuess(
    1,
    NL,
    -5.8,
    5.8,
    pole_dist=0.1,
    fact=1.0,
    cutoff=0.1,
    tol=1.0,
    decimals=3,
)
init_E = [E1[None,:], E2[None,:]]
init_G = [G1[None,:], G2[None,:]]
alpha_PO = 0.000
min_tol = np.zeros(NL)
min_tol1, min_tol2, min_tol3 = min_tol.copy(), min_tol.copy(), min_tol.copy()
sb1 = {}
eps = 1e-6
tol_ele = 1e-10
# In[]
def run_mini(its, method, which_e=None):
    R.Fit(
        fact=0.6,  # Redundant when we give init_E and init_G
        Fallback_W=5.0,  # Redundant\n",
        NumL=NL,  # Not redundant\n",
        fit_mode="all",  # Important to choose mode\n",
        force_PSD=True,  # the self-energies are not positive semidefinite
        force_PSD_tol=[min_tol1, min_tol2],  #
        use_analytical_jac=True,  # Important for speed
        min_method=method,  # Choose from any scipy.optimize.minimize method
        ebounds=(-7.5, 7.5),  # bounds on centres
        wbounds=(0.001, 5),  # bounds on widths
        gbounds=(None, None),  # bounds on sizes, redundant right now
        tol=tol_ele,  # any negative value with mean we fit all matrix elements of \\Gamma
        options={
            "disp": True,  # Minimizer options
            "maxiter": its,
            "gtol": 1e-10,
            #           'iprint':1,
        },
        #specific_bounds=[sb1, sb1],
        fit_real_part=False,  #
        alpha_PO=alpha_PO,  # Repulsion
        init_E=init_E,  # Give initial ei's and gi's
        init_G=init_G,
        which_e=which_e,
    )
    C = find_correction(R)
    R.Renormalise_H(C)
run_mini(0,'COBYLA')
# R.tofile('Test2')



