import numpy as np
import os
from time import time
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

# Read in stuff, you probably dont need to change anything here
Adir             = name + '/Arrays/'
Arrs             = {}
files = os.listdir(Adir)
class load_dictionary:
    def __init__(self, dir):
        self.dir = dir
        self.timer = time
        self.times = [self.timer()]
    def __getitem__(self, x):
        self.times += [time()]
        if max(self.times) - min(self.times)>120:
            assert  1 == 0
        return np.load(self.dir+x+'.npy')

Arrs = load_dictionary(name+'/Arrays/')

#for f in files:
#    Arrs.update({f[:-4]: np.load(Adir+f)})
# hbar is currently doubly defined in the code, this should be equal to the hbar used in the TimedepependentTransport module. Bug fix for the future

steps_for_bondcurrent = 10    # ??

