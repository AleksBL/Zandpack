import numpy as np
import os
from time import time
from Zandpack.td_constants import hbar
from Zandpack.Loader import load_dictionary
name             = 'U_zgnr'
eps              = 1e-9        # RK45 error tolerance
t0, t1           = -80.0, 800.0  # Start and End
usesave          = False        # Should the code use a previously saved run?
LoadFromFull     = False       # Have you stiched the previous run results?
saveevery        = 50          # How often to write the current and DM to file
checkpoints      = [40.0]       # When to save full system state
save_checkpoints = True        # If to save full system state
print_timings    = False       # More verbose output from the code (MASTER.txt & Worker*.txt)
stepsize         = 0.1         # Initial stepsize. Overwritten when usesave=True
n_dm_compress    = 5
save_PI          = False

# Read in stuff, you probably dont need to change anything here
Arrs = load_dictionary(name+'/Arrays/')
#for f in files:
#    Arrs.update({f[:-4]: np.load(Adir+f)})
# hbar is currently doubly defined in the code, this should be equal to the hbar used in the TimedepependentTransport module. Bug fix for the future

steps_for_bondcurrent = 10    # ??

