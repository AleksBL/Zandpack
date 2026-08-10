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
line = np.linspace(-2, 2, 151) + 1j*1e-2
line = np.vstack((line,line))

c = sisl.geom.sc(lat_const,'H').tile(3,1).add_vacuum(10,1).add_vacuum(10,2)
c.set_nsc((3,1,1))

for i in range(c.no):
    c.atoms[i] = sisl.Atom('H', R = 2.0)

geom_dev = c.tile(7,0)
geom_em = c.copy()
geom_ep = c.copy().move(geom_dev.cell[0,:] - geom_em.cell[0,:])
geom_dev = geom_dev.remove([10])
plt.show()

sisl.plot(geom_dev); plt.axis('equal')

Test = TD_Transport([geom_em,geom_ep], geom_dev, kT_i = [0.05, 0.05])
Test.Make_Contour(line, 15, pole_mode = 'JieHu2011')

Test.Electrodes( semi_infs = ['-a1', '+a1'] )
Test.make_device(elec_inds = [[i for i in range(3)],[i + 17 for i in range(3)]])

elec = sisl.Hamiltonian(c)
elec.construct([[0.1, lat_const * 1.1], 
                [0  , t_elec         ] ])

Test.run_electrodes(fois_gras_H = [elec, elec])
dev_H = sisl.Hamiltonian(geom_dev)
dev_H.construct([[0.1, lat_const * 1.1], 
                 [0,   t_dev             ] ])

Test.run_device(fois_gras_H = dev_H)
Test.read_data()

def run_fit(its):
    Test.Fit(fact = 2.0,             # a number controlling the broadening on the initial guess
          NumL = 5,               # number of Lorentzians used for the fitting
          fit_mode      = 'all',        # We fit the self-energies with lorentzians
          force_PSD     = True,                # the self-energies are not positive semidefinite
          force_PSD_tol = [0.0, 0.0], # 
          min_method    = 'SLSQP',# minimization method used by SciPy
          ebounds = (-2,2),       # bounds on where the Lorentzian centers can be located
          wbounds = (0.1, 1.0), # bounds on the linewidths of the Lorentzians
          gbounds = (None, None), # bounds on the fitting coefficients
          options = {'disp':True, # options keywords passed to the SciPy minimizer. The the docs for more info
                     'maxiter':its,   
                     'gtol':1e-5, 
                     'iprint':5},
          specific_bounds = None,#[{(0 ,5) :[(-0.1, 0.1), (4,5)]}, {(0 ,5) :[(-0.1, 0.1), (4,5)]}], 
          alpha_PO = 0.001,
         )

run_fit(500)
Test.pickle('Test')




