import os
ATOL = float(os.environ.get('ZP_ATOL', 1e-8))   # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', 1e-8))
T0 = float(os.environ.get('ZP_T0', 0))   # fast mode, see run_zandpack_tests --fast
T1 = float(os.environ.get('ZP_T1', 20))
N  = int(os.environ.get('ZP_NEVAL', 200))

"""Run the scipy_ode propagator on the density-matrix-feedback device and store
the current and the occupations for compare.py.

scipy_ode hands `dH` the density matrix at each Runge-Kutta stage, so the
feedback is integrated exactly as in the MPI codes.
"""

from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from Zandpack.Loader import flexload
import lzma

name = 'synth_dm'
tol  = 1e-10                         # same cutoff as tofile() used in device.py

C = load(lzma.open('synth_dm.xz','rb'))

C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(tol)

# Start from exactly the state the MPI codes read.  This also picks up any
# refinement done by SCF/psinought, which rewrites these arrays -- without it
# the two sides would start from different density matrices.
A       = name + '/Arrays/'
C.sigma = flexload(A + 'DM_Ortho.npy')
C.xi    = flexload(A + 'xi.npz')
C.Ixi   = flexload(A + 'Ixi.npz')
C.get_propagation_quantities()       # rebuild psi/omega from xi/Ixi

C.Check_input_to_ODE()

from shared import dH, delta

# shortlib lives at the TestRUN root, whatever the depth of this file
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != '/' and not _os.path.isdir(_os.path.join(_root, 'shortlib')):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from shortlib.driver import propagate          # ZP_RHS / ZP_CHUNK

t, dm, jl = propagate(C, dH, delta, 0.0, np.linspace(T0, T1, N) + 1e-5, ATOL, RTOL)

np.savez_compressed('scipy_ode_results.npz',
                    t  = t,
                    jl = jl,
                    N  = np.trace(dm, axis1 = -2, axis2 = -1).real)

plt.plot(t, jl.sum(axis = 1))
plt.xlabel('time')
plt.ylabel(r'$J_l$')
plt.savefig('scipy_ode.png', dpi = 300)
plt.close()
