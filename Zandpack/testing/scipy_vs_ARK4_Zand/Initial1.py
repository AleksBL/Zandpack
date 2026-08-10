import numpy as np
import os

name          = 'propfile'

eps              = 1e-5

t0, t1           = 0.0, 12.0  # Start and End
usesave          = False        # Should the code use a previously saved run?
LoadFromFull     = False       # Have you stiched the previous run results?
saveevery        = 50          # How often to write the current and DM to file
checkpoints      = [11.5]      # When to save full system state
save_checkpoints = True       # If to save full system state

print_timings    = False       # More verbose output from the code
stepsize         = 0.01         # Initial stepsize. Overwritten when usesave=True

Adir = name + '/Arrays/'
Arrs = {}
files = os.listdir(Adir)
for f in files:
    Arrs.update({f[:-4]: np.load(Adir+f)})

hbar = 6.582119569*10**-1     # electronvolt * femtosecond
steps_for_bondcurrent = 10    # ?? 


