import numpy as np
from Zandpack.Loader import load_dictionary
from Zandpack.td_constants import hbar

name          = 'TDZGNR'
eps           = 1e-8   # RK err. tol (SR lesson: keep the MPI side converged)
t0, t1        = 0, 20
usesave       = False
LoadFromFull  = False  # True if using the SCF-Psi state (steady pipeline)
saveevery     = 50     # How often to save curr. and RDM
checkpoints     = []
save_checkpoints = False
print_timings = False
stepsize      = 0.1
n_dm_compress = 1
save_PI       = False
steps_for_bondcurrent = 10

Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)

# fast mode: shorten the window from the environment (run_zandpack_tests --fast)
import os as _os
t0 = float(_os.environ.get("ZP_T0", t0))
t1 = float(_os.environ.get("ZP_T1", t1))
saveevery = int(_os.environ.get("ZP_SAVEEVERY", saveevery))
