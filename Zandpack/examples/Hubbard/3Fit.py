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
save = False
R = load(Open("Hub.xz", "rb"))
Eg = np.linspace(-5.9, 5.9, 250)
# R.reset_all_fits()
Name  = "AGNRTD"
NL    = 71
opts  = {'height': 1., 'distance': 5}
Emin, Emax = -7.5, 7.5
pdist = 0.1
fact  = 0.9
pfact = 0.8
cf    = 0.1
fm    = 'adaptive'
E01, G01 = R.PoleGuess(
    0,
    NL,
    Emin,
    Emax,
    fact=fact,
    cutoff=cf,
    tol=1.0,
    decimals=3,
    pole_dist = pdist,
    ik = 0,
    opts = opts,
    fillmode=fm,
    pole_fact = pfact
)
E11, G11 = R.PoleGuess(
    0,
    NL,
    Emin,
    Emax,
    fact=fact,
    cutoff=cf,
    tol=1.0,
    decimals=3,
    pole_dist = pdist,
    ik = 1,
    opts = opts,
    fillmode=fm,
    pole_fact = pfact
)

# Eh01 = np.array([-1.7, -1.5, -1.0, -0.5, 0.0, 1.2, 1.6, 2.6])
# Gh01 = np.array([0.18]*8)
# Eh02 = np.array([-2.5, -1.7, -1.1, -0.25, 0.0, 0.25, 0.5, 1.0])
# Gh02 = np.array([0.18]*8)
Eh01 = np.array([])
Gh01 = np.array([])
Eh02 = np.array([])
Gh02 = np.array([])


E1 = np.vstack((np.hstack((E01,Eh01)),np.hstack((E11, Eh02))))
G1 = np.vstack((np.hstack((G01,Gh01)),np.hstack((G11, Gh02))))

E02, G02 = R.PoleGuess(
    1,
    NL,
    Emin,
    Emax,
    fact=fact,
    cutoff=cf,
    tol=1.0,
    decimals=3,
    pole_dist = pdist,
    ik = 0,
    opts = opts,
    fillmode=fm,
    pole_fact = pfact
)
E12, G12 = R.PoleGuess(
    1,
    NL,
    Emin,
    Emax,
    fact=fact,
    cutoff=cf,
    tol=1.0,
    decimals=3,
    pole_dist = pdist,
    ik = 1,
    opts = opts,
    fillmode=fm,
    pole_fact = pfact
)

E2 = np.vstack((np.hstack((E02,Eh02)),np.hstack((E12, Eh01))))
G2 = np.vstack((np.hstack((G02,Gh02)),np.hstack((G12, Gh01))))
# E2 = np.vstack((E02,E12))
# G2 = np.vstack((G02,G12))

init_E = [E1, E2]
init_G = [G1, G2]
alpha_PO = 0.000
min_tol = np.zeros((NL))
min_tol[:] = -100000
min_tol1, min_tol2, min_tol3 = min_tol.copy(), min_tol.copy(), min_tol.copy()

sb1 = {}
eps = 1e-6
tol_ele = 1e-10
# In[]
def run_mini(its, method, which_e=None):
    R.Fit(
        fact=0.6,  # Redundant when we give init_E and init_G
        Fallback_W=5.0,  # Redundant\n",
        NumL=E1.shape[1],  # Not redundant\n",
        fit_mode="all",  # Important to choose mode\n",
        force_PSD=True,  # the self-energies are not positive semidefinite
        force_PSD_tol=[min_tol1, min_tol2],  #
        use_analytical_jac=True,  # Important for speed
        min_method=method,  # Choose from any scipy.optimize.minimize method
        ebounds=(-15.5, 15.5),  # bounds on centres
        wbounds=(0.001, 5),  # bounds on widths
        gbounds=(None, None),  # bounds on sizes, redundant right now
        tol=tol_ele,  # any negative value with mean we fit all matrix elements of \\Gamma
        options={
            "disp": True,  # Minimizer options
            "maxiter": its,
            "gtol": 1e-10,
            #           'iprint':1,
        },
        fit_real_part=False,  #
        alpha_PO=alpha_PO,  # Repulsion
        init_E=init_E,  # Give initial ei's and gi's
        init_G=init_G,
        which_e=which_e,
    )
run_mini(0,'COBYLA')
R.curvefit_all(.1)
R.NO_fitted_lorentzians[0].iterative_PSD(maxit=130, n=40,nW=15)
R.NO_fitted_lorentzians[1].iterative_PSD(maxit=130, n=40,nW=15)
R.FitNO2O()
C = find_correction(R, Emin = -1., Emax=1.)
R.Renormalise_H(C)
R.Inspect_transmission_from_hilbert_transform(E=Eg*1.5,eta=1e-2)



