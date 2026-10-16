#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:21:57 2022

@author: aleksander
"""

from TimedependentTransport.TimedependentTransport import TD_Transport
import sisl
import matplotlib.pyplot as plt
import numpy as np

t_dev =  1
t_elec = 1
lat_const = 1.0
line = np.linspace(-2, 2, 151) + 5j*1e-2
line = np.vstack((line,line))

g = sisl.geom.graphene(orthogonal = True)
geom_dev = g.tile(7,0)
geom_em  = g.tile(2,0)
geom_ep  = g.tile(2,0).move(5 * g.cell[0])
sisl.plot(geom_dev); plt.axis('equal')

Test = TD_Transport([geom_em,geom_ep], geom_dev, kT_i = [0.05, 0.05])
Test.Make_Contour(line, 15, pole_mode = 'JieHu2011')
Test.Electrodes ( kp = [[50,10,1],[50,10,1]],  )
Test.make_device( k = [1,10,1], k_tbtrans = [1,5,1] )
Test.run_electrodes()
Test.run_device()
Test.read_data(sub_orbital = [4 * i +2 for i in range(4 * 3)])

# In[]
#Test.reset_all_fits()

def run_fit(its):
    Test.Fit(fact = 2.0,             # a number controlling the broadening on the initial guess
             NumL = 5,               # number of Lorentzians used for the fitting
             fit_mode      = 'all',        # We fit the self-energies with lorentzians
             force_PSD     = True,                # the self-energies are not positive semidefinite
             force_PSD_tol = [0.0, 0.0], # 
             min_method    = 'SLSQP',# minimization method used by SciPy
             ebounds = (-3.5,3.5),       # bounds on where the Lorentzian centers can be located
             wbounds = (0.01, 2.0), # bounds on the linewidths of the Lorentzians
             gbounds = (None, None), # bounds on the fitting coefficients
             options = {'disp':True, # options keywords passed to the SciPy minimizer. The the docs for more info
                        'maxiter':its,   
                        'gtol':1e-5, 
                        'ftol':1e-5,
                        'iprint':1
                        },
             specific_bounds = None,#[{(0 ,5) :[(-0.1, 0.1), (4,5)]}, {(0 ,5) :[(-0.1, 0.1), (4,5)]}], 
             alpha_PO = 0.000,
             tol = 1e-2
             )


run_fit(100)
Test.pickle('Graw')




