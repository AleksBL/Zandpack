#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:53:07 2026

@author: aleks
"""

from Zandpack.wrapper import transiesta_hook, Control, Input
import numpy as np
recp_1 = """
# The bias function is not really called here
# the code is copied using the inspect.getsource
def bias(t,a):
    if a == 0: return np.sin(t)
    else:      return -np.sin(t)

init_file  = 'Output' # Name from when you saved your fit
work_name, work_dir  = 'ABC', 'TDcalc' # These names are determined by you
A = Input(work_name, t0=0.0, t1=50.0, eps=1e-10)
B = Control(A, init_file)
B.set_direc(work_dir)
B.init(overwrite=False)
# assumes Device.SiP is what the used SiP class is saved to.
S = transiesta_hook('Device.SiP', 'full')
B.set_hook(S)
# Write first, with full DFT dep
B.write_bias(bias=bias, hook = B.hook)
B.write_initial()
B.modify_occupation()
B.run_scf(DM_randomness=0.0)
B.run_psinought()
# set linearization flag for Mulliken linearization.
# makes hamiltonian evaluation much faster.
B.hook.scheme='lin_mul'
B.hook_linearize()
# update files and use nonorthogonal version (faster)
B.input.orthogonal=False
B.write_bias(bias=bias, hook = B.hook)
B.write_initial()
B.run_nozand("mpirun -np 3 ")
"""
