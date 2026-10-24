import os
T0 = float(os.environ.get('ZP_T0', 0))   # fast mode, see run_zandpack_tests --fast
T1 = float(os.environ.get('ZP_T1', 60))
N  = int(os.environ.get('ZP_NEVAL', 200))

from Zandpack.decoupled_functions import AdaptiveRK4 as RK4
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
f     = C.make_f()
sig   = C.sigma
psi   = C.Psi_vec
omega = C.omegas

tb, db = RK4(f, sig, psi, omega, 1e-10, -0, T1, dH, delta, C.Ixi,
             0.1, fixed_mode = False, name = 'Chain')

np.savez_compressed('adap_RK4_results.npz',
                    t  = tb,
                    jl = db['current_left'],
                    jr = db['current_right'],
                    N  = np.trace(db['density matrix'], axis1 = -2, axis2 = -1).real)

plt.plot(tb, np.array(db['current_left']))
plt.xlabel('time')
plt.ylabel(r'$J_l$')
plt.savefig('adap_RK4.png', dpi = 300)
plt.close()
