#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence sweep for the synthetic test.

Answers "how much of the scipy_ode - zand/nozand difference is just solver
accuracy?" by varying the settings the scripts set:

  * the Python side  : scipy_ode's atol/rtol (no MPI re-run needed)
  * the MPI side     : Initial.py's eps and stepsize (re-runs zand + nozand)

The metric is max|dJ| of the total left current, against the zand/nozand run,
on the overlapping time window.

    python sweep.py          # Python side only
    python sweep.py mpi      # also vary Initial.py eps/stepsize
"""

import os
import shutil
import subprocess
import sys
import lzma
from pickle import load

import numpy as np
from Zandpack.decoupled_functions import scipy_ode
from Zandpack.plot import J

TOL      = 1e-10                        # tofile / get_dense_matrices cutoff
N_EVAL   = 100
T_END    = 20
MPI      = len(sys.argv) > 1 and sys.argv[1] == 'mpi'

# from shared.py, so the sweep drives the device exactly like run.sh does
from shared import dH, delta

C = load(lzma.open('synth.xz','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(TOL)
C.Check_input_to_ODE()
f = C.make_f()


def total(j):
    return np.asarray(j).reshape(len(j), -1).sum(axis = 1)


def maxdiff(t_ref, y_ref, t_cmp, y_cmp):
    m = (t_cmp >= t_ref.min()) & (t_cmp <= t_ref.max())
    return np.abs(np.interp(t_cmp[m], t_ref, y_ref) - y_cmp[m]).max()


def scipy_run(atol, rtol, n_eval = N_EVAL):
    t, dm, jl = scipy_ode(f, C.sigma, C.Psi_vec, C.omegas, 0.0,
                          np.linspace(0, T_END, n_eval) + 1e-5, dH, delta, C.Ixi,
                          dH_given = True, method = 'RK45', dt_guess = None,
                          atol = atol, rtol = rtol)
    return t, jl.sum(axis = 1)


def mpi_run(dirs):
    """J() stitches several save dirs together, so pass exactly one here."""
    t, j = J(dirs)
    return t, total(j[0])


# ---- Python side ---------------------------------------------------------
t_ref, j_ref = mpi_run(['synth_save_zand'])
print('\nscipy_ode accuracy vs the (fixed) MPI run, %d eval points' % N_EVAL)
print('%10s %10s %12s' % ('atol', 'rtol', 'max|dJ|'))
for atol, rtol in [(1e-6, 1e-4), (1e-8, 1e-8), (1e-10, 1e-10), (1e-12, 1e-12)]:
    t, j = scipy_run(atol, rtol)
    print('%10.0e %10.0e %12.4g' % (atol, rtol, maxdiff(t_ref, j_ref, t, j)))

# The MPI codes write the current on their own (adaptive) time grid, while
# scipy_ode is only sampled at the eval points -- a coarse grid smooths the
# peaks away and under-reports the difference.
print('\neval-grid sensitivity (atol = rtol = 1e-10), vs the dense MPI current')
print('%10s %12s' % ('n_eval', 'max|dJ|'))
for n in (100, 400, 2000):
    t, j = scipy_run(1e-10, 1e-10, n)
    print('%10d %12.4g' % (n, maxdiff(t_ref, j_ref, t, j)))

# ---- MPI side ------------------------------------------------------------
if MPI:
    t_ref, j_ref = scipy_run(1e-12, 1e-12)          # tightest Python reference
    print('\nMPI accuracy vs scipy_ode(1e-12), varying Initial.py')
    print('%10s %10s %8s %12s' % ('eps', 'stepsize', 'code', 'max|dJ|'))
    shutil.copy('Initial.py', '.Initial_backup.py')
    try:
        for eps, step in [(1e-10, 0.1), (1e-8, 0.1), (1e-8, 0.05)]:
            txt = open('.Initial_backup.py').read()
            txt = txt.replace('eps           = 1e-10', 'eps           = %g' % eps)
            txt = txt.replace('stepsize      = 0.1',   'stepsize      = %g' % step)
            open('Initial.py','w').write(txt)
            for code, opt in (('zand',''), ('nozand','bias_file=Bias_NO.py')):
                subprocess.run('mpirun -np 3 %s Dir=$PWD %s > /dev/null 2>&1'
                               % (code, opt), shell = True, check = True)
                shutil.rmtree('sweep_%s' % code, ignore_errors = True)
                os.rename('synth_save', 'sweep_%s' % code)
            for code in ('zand', 'nozand'):
                t, j = mpi_run(['sweep_%s' % code])
                print('%10.0e %10.2f %8s %12.4g'
                      % (eps, step, code, maxdiff(t_ref, j_ref, t, j)))
                shutil.rmtree('sweep_%s' % code, ignore_errors = True)
    finally:
        shutil.move('.Initial_backup.py', 'Initial.py')
