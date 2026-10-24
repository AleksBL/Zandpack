import numpy as np
from Zandpack.Loader import load_dictionary
from Zandpack.td_constants import hbar

name          = 'propfile'

eps              = 1e-5

t0, t1           = 11.0, 20.0  # Start and End
usesave          = True        # Should the code use a previously saved run?
LoadFromFull     = False       # Have you stiched the previous run results?
saveevery        = 50          # How often to write the current and DM to file
checkpoints      = [11.1]      # When to save full system state
save_checkpoints = True       # If to save full system state

print_timings    = False       # More verbose output from the code
stepsize         = 0.01         # Initial stepsize. Overwritten when usesave=True

n_dm_compress    = 1
save_PI          = False

Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)

steps_for_bondcurrent = 10    # ??

# fast mode: shorten the window from the environment (run_zandpack_tests --fast)
import os as _os
t0 = float(_os.environ.get("ZP_T0", t0))
t1 = float(_os.environ.get("ZP_T1", t1))
saveevery = int(_os.environ.get("ZP_SAVEEVERY", saveevery))
