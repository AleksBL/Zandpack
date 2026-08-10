import numpy as np
import os
from Zandpack.Loader import load_dictionary
from Zandpack.td_constants import hbar
name          = 'TDT'

eps = 1e-12

t0, t1        = -3 * np.pi/(2 * 0.1)-100, 150
usesave       = False
print_timings = False
stepsize      = 0.1
saveevery     = 50
checkpoints      = [2.5]       # When to save full system state
save_checkpoints = False       # If to save full system state
LoadFromFull     = False       # Have you stiched the previous run results?
n_dm_compress = 2
save_PI = False
print_timings=False
Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)
steps_for_bondcurrent = 10

