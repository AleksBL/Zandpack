import numpy as np
import os
from time import time
from Zandpack.Loader import load_dictionary
from Zandpack.td_constants import hbar

name             = 'RSSE'
eps              = 1e-6        # RK45 error tolerance
t0, t1           = -100.0, 200.0  # Start and End
usesave          = True      # Should the code use a previously saved run?
LoadFromFull     = True       # Have you stiched the previous run results?
saveevery        = 50          # How often to write the current and DM to file
checkpoints      = [40.0]       # When to save full system state
save_checkpoints = True        # If to save full system state
print_timings    = False       # More verbose output from the code (MASTER.txt & Worker*.txt)
stepsize         = 0.1         # Initial stepsize. Overwritten when usesave=True
n_dm_compress    = 10
save_PI          = True

Adir             = name + '/Arrays/'
Arrs = load_dictionary(name+'/Arrays/')
steps_for_bondcurrent = 10    # ??


