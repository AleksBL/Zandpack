#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic test: the smallest device that still exercises the whole pipeline.

A chain of `nsite` sites with one orbital each, hand built with sisl -- no DFT,
no Siesta run, no k-point sampling.  The four numbers in the tuning block are
what controls the size of the problem, so the test is easy to rescale.

Matrix shapes produced here (nk = 1 throughout):
    H, sigma        (nk, no, no)
    xi, Ixi         (nk, nlead, Nm, noT, nc)
    omegas          packed, see Zandpack.decoupled_functions.make_f
"""

from Zandpack.TimedependentTransport import TD_Transport
import sisl
import numpy as np

# ---- tuning block ---------------------------------------------------------
nsite     = 4          # device sites (one orbital each => no = nsite)
t_hop     = 1.0        # nearest neighbour hopping
e_onsite  = 0.1        # on-site energy
kT        = 0.05       # electronic temperature
n_contour = 41         # energy pointS on the contour
N_F       = 2          # Fermi poles
NumL      = 2          # Lorentzians used in the fit
tol       = 1e-10      # shared tofile / get_dense_matrices cutoff
# --------------------------------------------------------------------------

lat_const = 1.0
line = np.vstack([np.linspace(-2, 2, n_contour) + 1j*1e-2]*2)

c        = sisl.geom.sc(lat_const, sisl.Atom('H', R = 2.0)).add_vacuum(6, 1).add_vacuum(6, 2)
geom_dev = c.tile(nsite, 0)
geom_em  = c.copy()
geom_ep  = c.copy().move(geom_dev.cell[0,:] - geom_em.cell[0,:])

Test = TD_Transport([geom_em, geom_ep], geom_dev, kT_i = [kT, kT])
Test.Make_Contour(line, N_F, pole_mode = 'JieHu2011')

Test.Electrodes( semi_infs = ['-a1', '+a1'] )
Test.make_device(elec_inds = [[0], [nsite - 1]])

elec = sisl.Hamiltonian(c)
elec.set_nsc((3,1,1))                 # the electrode is periodic along a1
elec.construct([[e_onsite, lat_const * 1.1],
                [0,        t_hop             ]])

dev_H = sisl.Hamiltonian(geom_dev)
dev_H.set_nsc((1,1,1))                # no periodic couplings between electrodes
dev_H.construct([[e_onsite, lat_const * 1.1],
                 [0,        t_hop             ]])

Test.run_electrodes(manual_H = [elec, elec])
Test.run_device(manual_H = dev_H)
Test.read_data()

Test.Fit(fact = 1.0, NumL = NumL, fit_mode = 'all',
         force_PSD = True, force_PSD_tol = [0.0, 0.0],
         min_method = 'SLSQP', ebounds = (-2, 2), wbounds = (0.01, 1.0),
         options = {'maxiter': 0}, alpha_PO = 0.0)

Test.pickle('synth')
Test.tofile('synth', tol = tol)
