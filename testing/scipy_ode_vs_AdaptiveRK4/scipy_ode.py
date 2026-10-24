import os
ATOL = float(os.environ.get('ZP_ATOL', 1e-7))   # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', 1e-5))
T0 = float(os.environ.get('ZP_T0', 0))   # fast mode, see run_zandpack_tests --fast
T1 = float(os.environ.get('ZP_T1', 60))
N  = int(os.environ.get('ZP_NEVAL', 200))

import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from shared import dH, delta
import lzma


C = load(lzma.open('Test.xz','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(1e-10)

# Start from the same state the MPI codes read; this also picks up any
# refinement done by SCF/psinought, which rewrites these arrays.
from Zandpack.Loader import flexload
A       = 'TDT_chain/Arrays/'
C.sigma = flexload(A + 'DM_Ortho.npy')
C.xi    = flexload(A + 'xi.npz')
C.Ixi   = flexload(A + 'Ixi.npz')
C.get_propagation_quantities()      # rebuild psi/omega from xi/Ixi
C.Check_input_to_ODE()

# shortlib lives at the TestRUN root, whatever the depth of this file
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != '/' and not _os.path.isdir(_os.path.join(_root, 'shortlib')):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from shortlib.driver import propagate          # ZP_RHS / ZP_CHUNK

t, dm, jl = propagate(C, dH, delta, 0.0, np.linspace(T0,T1,N)+1e-5, ATOL, RTOL)

np.savez_compressed('scipy_ode_results.npz', t = t, jl = jl,
                    N = np.trace(dm, axis1 = -2, axis2 = -1).real)

plt.plot(t, jl.sum(axis = 1))
plt.xlabel('time')
plt.ylabel(r'$J_l$')
plt.savefig('scipy_ode.png', dpi = 300)
plt.close()
