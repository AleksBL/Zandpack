import numpy as np
import os
from Zandpack.td_constants import hbar
from Zandpack.Loader import load_dictionary
name          = 'TDT_Croy2009'
eps = 1e-10
t0, t1        = -10, 40
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
steps_for_bondcurrent = 10
Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)
