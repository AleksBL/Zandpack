#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic test whose dH depends on the density matrix.

The device is a tiny chain (no DFT, no k-point sampling, two propagated
orbitals), but -- unlike synth_1 -- it is **non-orthogonal**: the nearest
neighbour overlap is set explicitly, so the Loewdin transforms are not the
identity.  That matters here, because the driving field feeds back on the
density matrix: a mean-field (Hartree-like) term makes the potential on each end
site follow the charge on the other end, `out[0,i,i] = Rep * sigma[0,j,j]`,
which is the pattern `examples/ThesisExample/Bias.py` uses for its ZGNR
repulsion.

With `S != I` the nonorthogonal (nozand) run has to convert the density matrix
*before* evaluating dH, not just the result -- see `Bias_NO.py`.  On an
orthogonal device that step degenerates to a no-op and cannot be checked.

The feedback makes the equations of motion **non-linear**, which no other test
covers.  `dH` is evaluated at the current Runge-Kutta stage by every method (the
Python `make_f` and both MPI codes pass the stage density matrix), so the four
propagators are still being asked the identical question.

Everything that sets the size sits in the tuning block here (device) and at the
top of `shared.py` (drive + feedback).
"""

from Zandpack.TimedependentTransport import TD_Transport
import sisl
import numpy as np

# ---- tuning block ---------------------------------------------------------
nsite     = 4          # device sites (one orbital each); sites 0 and nsite-1
                       # become the electrodes, so 2 orbitals are propagated
t_hop     = 1.0        # nearest neighbour hopping
s_hop     = 0.2        # nearest neighbour overlap -- this is what makes S != I
e_onsite  = 0.1        # on-site energy
kT        = 0.05       # electronic temperature
n_contour = 41         # energy points on the contour
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

def build(self, ia, atoms, atoms_xyz = None):
    """On-site energy + nearest neighbour hopping, and the matching overlap.

    `s_hop` is what makes the device non-orthogonal, so `S != I` and the Loewdin
    transforms used by the nozand run are not the identity.
    """
    idx = self.geometry.close(ia, R = [0.5, lat_const * 1.05],
                              atoms = atoms, atoms_xyz = atoms_xyz)
    self[ia, idx[0]]   = e_onsite
    self[ia, idx[1]]   = t_hop
    self.S[ia, idx[0]] = 1.0
    self.S[ia, idx[1]] = s_hop


elec  = sisl.Hamiltonian(c, orthogonal = False)
elec.set_nsc((3,1,1))                 # the electrode is periodic along a1
elec.construct(build)

dev_H = sisl.Hamiltonian(geom_dev, orthogonal = False)
dev_H.set_nsc((1,1,1))                # no periodic couplings between electrodes
dev_H.construct(build)

Test.run_electrodes(manual_H = [elec, elec])
Test.run_device(manual_H = dev_H)
Test.read_data()

Test.Fit(fact = 1.0, NumL = NumL, fit_mode = 'all',
         force_PSD = True, force_PSD_tol = [0.0, 0.0],
         min_method = 'SLSQP', ebounds = (-2, 2), wbounds = (0.01, 1.0),
         options = {'maxiter': 0}, alpha_PO = 0.0)

Test.pickle('synth_dm')
Test.tofile('synth_dm', tol = tol)
