import numpy as np
import os

name          = 'TDT_Chain?'

eps =  1e-10

t0, t1        = 0, 60
usesave       = False
print_timings = False
stepsize      = 0.1
saveevery     = 50

Adir = name + '/Arrays/'
Arrs = {}
files = os.listdir(Adir)
for f in files:
    Arrs.update({f[:-4]: np.load(Adir+f)})
    
hbar = 6.582119569*10**-1
steps_for_bondcurrent = 10

