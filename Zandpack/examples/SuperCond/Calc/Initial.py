import numpy as np
#import os
from time import time
from Zandpack.td_constants import hbar
from Zandpack.Loader import load_dictionary
name             = 'SCBenzene_2'#os.environ['SAVEFOLDER']
loadname         = 'SCBenzene_2'#os.environ['LOADFOLDER']
eps              = 1e-8        # RK45 error tolerance
t0, t1           = -80.0, 50.0  # Start and End
usesave          = True        # Should the code use a previously saved run?
LoadFromFull     = True      # Have you stiched the previous run results?
saveevery        = 50          # How often to write the current and DM to file
checkpoints      = [40.0]      # When to save full system state
save_checkpoints = True        # If to save full system state
print_timings    = False       # More verbose output from the code (MASTER.txt & Worker*.txt)
stepsize         = 0.1         # Initial stepsize. Overwritten when usesave=True
n_dm_compress    = 2           # Doesnt save DM at all step only every n_dm_compress'th
save_PI          = True       # Save PI matrix?
steps_for_bondcurrent = 10     # ??

# Read in stuff, you probably dont need to change anything here
Arrs = load_dictionary(loadname+'/Arrays/')




