#!/usr/bin/env python3
"""
Run the decoupled SciPy ODE propagator on the TDZGNR device and store the
resulting current and occupations so they can be compared against the
zand/nozand runs.
"""

import os
# Default tolerances are looser than the other tests (1e-8/1e-8): this device's
# fitted self-energies have fast Lorentzian modes (max Gamma eigenvalue ~265),
# and at 1e-8 solve_ivp takes ~1e-6-sized steps whose per-step state retention
# grows past 30 GB.  At 1e-6/1e-4 the pointwise agreement with zand (eps=1e-8)
# is 2.1e-04 on t = 0..2 -- at the level of the other tests.
ATOL = float(os.environ.get('ZP_ATOL', 1e-6))   # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', 1e-4))
T0 = float(os.environ.get('ZP_T0', 0))   # fast mode, see run_zandpack_tests --fast
T1 = float(os.environ.get('ZP_T1', 20))
N  = int(os.environ.get('ZP_NEVAL', 200))

# This device's propagation drifts past 30 GB on the package's scipy_ode path
# (chunk=1: one solve_ivp per eval point, and its memory grows with the
# window).  The piecewise method (ZP_CHUNK > 1) carries the state across
# pieces and keeps only the current, which bounds the peak; 10 eval points per
# call keeps it around a few GB here.  See shortlib/README.md.
os.environ.setdefault('ZP_CHUNK', '10')

from pickle import load
import numpy as np
import lzma
import matplotlib.pyplot as plt
from shared import dH, bias as delta

C = load(lzma.open('ZGNRfit.xz', 'rb'))

C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(1e-4)     # the same tol as tofile in device.py

# Start from the same state the MPI codes read; this also picks up any
# refinement done by SCF/psinought, which rewrites these arrays.
from Zandpack.Loader import flexload
A       = 'TDZGNR/Arrays/'
C.sigma = flexload(A + 'DM_Ortho.npy')
C.xi    = flexload(A + 'xi.npz')
C.Ixi   = flexload(A + 'Ixi.npz')
C.get_propagation_quantities()      # rebuild psi/omega from xi/Ixi
C.Check_input_to_ODE(loose_fermi = True)

# shortlib lives at the TestRUN root, whatever the depth of this file
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != '/' and not _os.path.isdir(_os.path.join(_root, 'shortlib')):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from shortlib.driver import propagate          # ZP_RHS / ZP_CHUNK

t, dm, jl = propagate(C, dH, delta, 0.0, np.linspace(T0, T1, N) + 1e-5, ATOL, RTOL)

# jl holds one column per k-point, the total current is their sum.
# dm is only kept for the first k-point by scipy_ode, so N is a single curve.
np.savez_compressed('scipy_ode_results.npz',
                    t  = t,
                    jl = jl,
                    N  = np.trace(dm, axis1 = -2, axis2 = -1).real)

plt.plot(t, jl.sum(axis = 1))
plt.xlabel('time')
plt.ylabel(r'$J_l$')
plt.savefig('scipy_ode.png', dpi = 300)
plt.close()
