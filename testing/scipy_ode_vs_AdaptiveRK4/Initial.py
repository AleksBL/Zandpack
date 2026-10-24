import numpy as np
from Zandpack.Loader import load_dictionary
from Zandpack.td_constants import hbar

name          = 'TDT_chain'

eps           = 1e-10
t0, t1        = 0, 60
usesave       = False
LoadFromFull  = False       # Have you stiched the previous run results?
print_timings = False
stepsize      = 0.1
saveevery     = 50          # How often to write the current and DM to file
checkpoints      = []       # When to save full system state
save_checkpoints = False    # If to save full system state
n_dm_compress = 1
save_PI       = False

Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)

steps_for_bondcurrent = 10

# fast mode: shorten the window from the environment (run_zandpack_tests --fast)
import os as _os
t0 = float(_os.environ.get("ZP_T0", t0))
t1 = float(_os.environ.get("ZP_T1", t1))
saveevery = int(_os.environ.get("ZP_SAVEEVERY", saveevery))
