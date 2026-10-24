import numpy as np
import os
from Zandpack.Loader import load_dictionary
from Zandpack.td_constants import hbar
name          = 'TDT_Croy2016'

eps = 1e-8

t0, t1        = 0, 200
usesave       = False
print_timings = False
stepsize      = 0.1

LoadFromFull     = False       # Have you stiched the previous run results?
saveevery        = 50          # How often to write the current and DM to file
checkpoints      = [2.5]       # When to save full system state
save_checkpoints = False       # If to save full system state

n_dm_compress = 2
save_PI = False
print_timings=False
Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)
steps_for_bondcurrent = 10


# fast mode: shorten the window from the environment (run_zandpack_tests --fast)
import os as _os
t0 = float(_os.environ.get("ZP_T0", t0))
t1 = float(_os.environ.get("ZP_T1", t1))
saveevery = int(_os.environ.get("ZP_SAVEEVERY", saveevery))
