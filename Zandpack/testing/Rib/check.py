#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:00:13 2023

@author: aleksander
"""

from pickle import load
import numpy as np
from Zandpack.plot import plt
from lzma import open
from Zandpack.TimedependentTransport import AdaptiveRK4  as RK4
from Zandpack.TimedependentTransport import scipy_ode
from Zandpack.Pulses import box_pulse
from numba import njit

# In[]
C = load(open('Fit_2.xz','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(zero_tol=1e-4)
C.Check_input_to_ODE()
f     = C.make_f()
sig   = C.sigma
psi   = C.Psi_vec
omega = C.omegas

@njit
def bias(t,a):
    if a == 0:
        return -0.5 + (np.cos(t) + np.cos(1.5*t)) * box_pulse(t+20,40,.5, 1.0)
    if a == 1:
        return 0.5 - (np.cos(t/3) + np.cos(t)) * box_pulse(t+20,40,.5, 1.0)

@njit
def dH(t,sig):
    o  = np.zeros(sig.shape, dtype=np.complex128)
    o[0,20,20] += 2.0*o[0,20,20]
    return o

#ts, ds   =  RK4(f, sig,psi, omega, 1e-5, 0, 40, dH, bias, C.Ixi, name='stm')
t1,dm,jl =  scipy_ode(f, sig, psi, omega, 0.0, np.linspace(0,50,100)+1e-5, dH, bias, C.Ixi, 
                      dH_given = True, method = 'RK45', dt_guess = None, 
                      atol = 1e-8, rtol = 1e-6)
