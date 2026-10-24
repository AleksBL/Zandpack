#!/usr/bin/env python3
"""ThesisZGNR test device: a 3-ZGNR from SIESTA (SZ basis), TranSIESTA +
TBtrans, pz-only device orbitals, Lorentzian fit of the self-energies.

Adapted from examples/ThesisExample (5a/5b/5d).  The steady-state preparation
(TDeq_1.sh: ts-contour, modify_occupations, SCF, psinought) is deliberately
NOT part of this script -- like every test in the suite, the default run
starts from the Fermi density matrix of the device Hamiltonian; an SCF +
psinought pipeline belongs in a separate run_steady.sh.

The fit uses 25 Lorentzians per lead where the thesis example used 81, to keep
the auxiliary-mode count (and so the propagation cost) at test size.
"""

from siesta_python.siesta_python import SiP
from Zandpack.TimedependentTransport import TD_Transport as TDT
from Zandpack.FittingTools import find_correction
import sisl
import numpy as np

# ---- tuning block ---------------------------------------------------------
tx_e, tx_d, kp = 2, 12, [30, 1, 1]   # electrode/device lengths, lead k-grid
N_F  = 5        # Fermi poles on the contour
NL   = 25       # Lorentzians per lead (ThesisExample used 81)
tol  = 1e-4     # tofile / zero_tol cutoff, the same on both sides
# ---------------------------------------------------------------------------

def run(A):
    A.Passivate(1, 1.1)
    if len(A.elecs) != 0: A.find_elec_inds()
    A.fdf(); A.run_siesta_in_dir()

# Geometry: 3-ZGNR, electrodes 2 cells, device 12 cells with 3 sites removed
g   = sisl.geom.zgnr(3)
gem = g.tile(tx_e, 0)
gep = gem.move(g.cell[0] * (tx_d - tx_e))
gd  = g.tile(tx_d, 0).remove([35, 38, 41])

EM = SiP(gem.cell, gem.xyz, gem.atoms.Z, mpi = '', directory_name = 'EM',
         sl = 'EM', kp = kp, semi_inf = '-a1', basis = 'SZ')
EP = SiP(gep.cell, gep.xyz, gep.atoms.Z, mpi = '', directory_name = 'EP',
         sl = 'EP', kp = kp, semi_inf = '+a1', basis = 'SZ')
Dev = SiP(gd.cell, gd.xyz, gd.atoms.Z, directory_name = 'Dev',
          elecs = [EM, EP], basis = 'SZ', Chem_Pot = [0, 0], mpi = '',
          solution_method = 'transiesta', save_SE = True)
run(EM); run(EP)

# TBtrans' "electrode connectivity is not perfect" check: the electrode TSHS
# must carry the full coupling range along the transport direction, so rewrite
# it with nsc = (3,1,1) (read_TSHS -> set_nsc -> manual_H, which writes the
# TSHS back).
for E in (EM, EP):
    Hm = E.read_TSHS()
    Hm.set_nsc((3, 1, 1))
    E.manual_H(Hm)

run(Dev)

# TD_Transport object on the SIESTA output (5b)
gd  = Dev.to_sisl()
ge1, ge2 = Dev.elecs[0].to_sisl(), Dev.elecs[1].to_sisl()
R   = TDT([ge1, ge2], gd, kT_i = [.025, .025], mu_i = [.0, .0])

line = np.linspace(-8, 10, 201) + 1e-2j + 1e-3
line = np.vstack((line, line))
R.Make_Contour(line, N_F)
Dev.custom_tbtrans_contour = R.Contour
Dev.fdf(eta = 1.0e-2)
Dev.run_tbtrans_in_dir()
R.Device = Dev

# Extract the p_z part of the carbon orbitals (cC = 4 orbitals/C, index 2)
cC, cH, it = 4, 1, 0
sub_idx = []
for i in range(len(Dev.s)):
    if Dev.s[i] == 6:
        sub_idx += [it + 2]
        it += cC
    elif Dev.s[i] == 1:
        it += cH
print('device orbitals after pz selection: %d' % len(sub_idx))

R.read_data(sub_orbital = sub_idx)

# Fit (5d, with NL = 25 instead of 81)
opts  = {'height': 1.0, 'distance': 5}
pdist, fact, pfact, cf = 0.1, 0.9, 0.8, 0.1
E1, G1 = R.PoleGuess(0, NL, -8.0, 8.0, fact = fact, cutoff = cf, tol = 1.0,
                     decimals = 3, pole_dist = pdist, pole_fact = pfact, opts = opts)
E2, G2 = R.PoleGuess(1, NL, -8.0, 8.0, fact = fact, cutoff = cf, tol = 1.0,
                     decimals = 3, pole_dist = pdist, pole_fact = pfact, opts = opts)
min_tol = -100000 * np.ones(NL)[None, :]
R.Fit(fact = 1.0, NumL = NL,
      force_PSD_tol = [min_tol, min_tol],
      min_method = 'COBYLA',
      ebounds = (-15., 15.), wbounds = (0.001, 5),
      tol = 1e-3,
      options = {'maxiter': 0, 'gtol': 1e-5},
      fit_real_part = False, alpha_PO = 0.0,
      init_E = [E1[None, :], E2[None, :]],
      init_G = [G1[None, :], G2[None, :]])
R.curvefit_all(0.0001)
R.NO_fitted_lorentzians[0].iterative_PSD(maxit = 100, n = 40, nW = 15,
                                         lbtol = -0.001, fact = .5, add_last = False)
R.NO_fitted_lorentzians[1].iterative_PSD(maxit = 100, n = 40, nW = 15,
                                         lbtol = -0.001, fact = .5, add_last = False)
R.FitNO2O()
C = find_correction(R, Emin = -1, Emax = 1)
R.Renormalise_H(C)

R.pickle('ZGNRfit')
R.tofile('TDZGNR', tol = tol)

# ----------------------------------------------------------------------------
# NOTE ON THE FITTED SELF-ENERGIES (read this before trusting the physics):
# the Lorentzian fit can produce Gamma matrices with NEGATIVE eigenvalues
# ("If the minimum is negative, you should take extra care!" -- printed by the
# fit above; the 25-pole fit here has min eig ~ -13).  This test's purpose is
# to check that shortlib/scipy_ode, zand and nozand agree on the SAME fitted
# problem, which they do -- every method consumes the identical Arrays/.  But
# a non-PSD fit is not a trustworthy self-energy: physical quantities
# (currents, occupations, transients) may be distorted by it, and a growing
# mode from a negative eigenvalue can make any propagator take very small
# steps.  See examples/ThesisExample, which shows the same behaviour with its
# 81-pole fit.
# ----------------------------------------------------------------------------
