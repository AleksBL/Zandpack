#!/usr/bin/env python3
from Zandpack.wrapper import transiesta_hook, Control, Input
import numpy as np
# This function is never called, but is written to the Bias.py
# file using the inspect module
def bias(t,a):
    if a == 0: return 0.0 # np.sin(t)
    else:      return 0.0 # -np.sin(t)
# The output from the tofile function called at the end of the fitting procedure
init_file  = 'YourOutputFromFit'
work_name, work_dir  = 'ABC', 'TDcalc'
A = Input(work_name, t0=0.0, t1=50.0, eps=1e-8)
B = Control(A, init_file)
B.set_direc(work_dir)
B.init(overwrite=True)
S = transiesta_hook('Device.SiP', 'full')
B.set_hook(S, Rmax=15.0)
# Write first, with full DFT dep
B.write_bias(bias=bias, hook=B.hook)
B.write_initial()
# Depending on the amount of unoccupied states, you might
# have to increase the number of poles for the Fermi function
# expansion.
B.modify_occupation(eigtol=1e-3, N_F=30, kT_i=[0.05, 0.05])
B.run_scf(DM_randomness=0.0, write_dm_every=10, weight=0.25,
          DM_start_file="../SCFintermediateDM.npy",
          # Contour="../mycontour_2.npy",
          write_progress=True)
# Checks if the eigenvalues of the last Ham. in the
# SCF cycle has eigenvalues within the bandwidth of
# the pole expanded fermi function.
B.check()
# Set linearization flag to linearization in orthogonal DM diagonal elements.
# This makes hamiltonian evaluation much faster.
B.hook.scheme='lin_odm'
B.hook_linearize()
B.write_bias(bias=bias, hook=B.hook)
B.write_initial()
B.run_scf(DM_randomness=0.0, write_dm_every=10, weight=0.25,
          # DM_start_file="../SCFintermediateDM.npy",
          #Contour="../mycontour_2.npy",
          write_progress=True)
B.run_psinought()
# update files and use nonorthogonal version (faster)
B.input.orthogonal=False
B.write_bias(bias=bias, hook=B.hook)
B.write_initial()
B.run_nozand("mpirun -np 3 ")
