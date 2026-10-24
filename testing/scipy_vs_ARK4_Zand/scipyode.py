import os
ATOL = float(os.environ.get('ZP_ATOL', 1e-8))   # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', 1e-8))
T0 = float(os.environ.get('ZP_T0', 0))   # fast mode, see run_zandpack_tests --fast
# scipy_ode's peak memory grows super-linearly with the simulated window on this
# device (it runs past 20 GB over t = 0..2, whatever the eval grid), so the stock
# window is short and has to match Initial.py.  This test originally ran 0 .. 20;
# raise T1/N here and t1 in Initial.py together, only if you have the RAM.
T1 = float(os.environ.get('ZP_T1', 1.0))
N  = int(os.environ.get('ZP_NEVAL', 10))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:50:05 2022

@author: aleksander
"""

import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from shared import dH, delta
import lzma

R = load(lzma.open('DevFit.xz','rb'))
R.diagonalise()
R.get_propagation_quantities()
R.get_dense_matrices(zero_tol = 1e-5)

# Start from the same state the MPI codes read; this also picks up any
# refinement done by SCF/psinought, which rewrites these arrays.
from Zandpack.Loader import flexload
A       = 'propfile/Arrays/'
R.sigma = flexload(A + 'DM_Ortho.npy')
R.xi    = flexload(A + 'xi.npz')
R.Ixi   = flexload(A + 'Ixi.npz')
R.get_propagation_quantities()      # rebuild psi/omega from xi/Ixi
R.Check_input_to_ODE()

# shortlib lives at the TestRUN root, whatever the depth of this file
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != '/' and not _os.path.isdir(_os.path.join(_root, 'shortlib')):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from shortlib.driver import propagate          # ZP_RHS / ZP_CHUNK

t,dm,jl = propagate(R, dH, delta, 0.0, np.linspace(T0,T1,N)+1e-7, ATOL, RTOL)

plt.plot(t, jl)
plt.xlabel('time')
plt.ylabel(r'$J_l$')
plt.savefig('scipy_ode.png', dpi = 300)
plt.close()
np.savez('ScipyODE', t = t, dm = dm, jl = jl,
         N = np.trace(dm, axis1 = -2, axis2 = -1).real)
