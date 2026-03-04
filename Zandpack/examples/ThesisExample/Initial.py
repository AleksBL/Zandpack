import numpy as np
from time import time
from Zandpack.td_constants import hbar
from Zandpack.Loader import load_dictionary
name            = 'TDZGNR'
eps             = 1e-5  #RK err. tol
t0, t1          = -25.0, 100# start and end time
usesave     = True      #continuation run? (inc. SCF-Psi)
LoadFromFull= True      #True if using SCF-Psi state
saveevery   = 50        #How often to save curr. and RDM
checkpoints     = [40.0]#When to save full system state
save_checkpoints= True  #If to save full system state
print_timings   = False #More verbose output
stepsize        = 0.1 #Initial stepsize. Superceeded by usesave=True
n_dm_compress   = 5     #Dont save all DMs, they take a lot of space.
save_PI         = False #Save Current matrix? 
steps_for_bondcurrent = 10
Arrs  = load_dictionary(name + '/Arrays/')#Read in arrays
