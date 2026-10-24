#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:00:13 2023

@author: aleksander

Compare scipy_ode against the zand and nozand MPI codes on the ribbon device.
dH and bias are imported from Bias.py, so the Python and the MPI runs use
exactly the same driving.

Left  panel: the current, read through Zandpack.plot.J for the MPI runs.
Right panel: Tr(DM) per k-point, read through Zandpack.plot.N; nozand is given
             the overlap S so it reports the same particle number as the
             orthogonal runs.
The difference metrics are printed as well, so the test can be read without
looking at the plots.
"""

from pickle import load
import numpy as np
from lzma import open
from Zandpack.plot import plt, J, N
from Zandpack.Help import TDHelper
from Zandpack.decoupled_functions import scipy_ode
from shared import dH, delta

C = load(open('Fit_5.xz','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(zero_tol = 1e-4)

# Start from the same state the MPI codes read; this also picks up any
# refinement done by SCF/psinought, which rewrites these arrays.
from Zandpack.Loader import flexload
A       = 'Rib_5/Arrays/'
C.sigma = flexload(A + 'DM_Ortho.npy')
C.xi    = flexload(A + 'xi.npz')
C.Ixi   = flexload(A + 'Ixi.npz')
C.get_propagation_quantities()      # rebuild psi/omega from xi/Ixi
C.Check_input_to_ODE()
f = C.make_f()

t0, dm, jl = scipy_ode(f, C.sigma, C.Psi_vec, C.omegas, 0.0, np.linspace(0, 40, 100) + 1e-5,
                       dH, delta, C.Ixi, dH_given = True, method = 'RK45',
                       dt_guess = None, atol = 1e-8, rtol = 1e-6)

sp_t  = t0
sp_jl = jl.sum(axis = 1)
sp_N  = np.trace(dm, axis1 = -2, axis2 = -1).real

t_z, jz = J(['Rib_5_save_zand'])
t_n, jn = J(['Rib_5_save_nozand'])

Hlp       = TDHelper('Rib_5')
t_Nz, N_z = N(['Rib_5_save_zand'])
t_Nn, N_n = N(['Rib_5_save_nozand'], S = Hlp.S, insert_tril = True)


def total(j):
    """zand/nozand report one current column per k-point; sum for the total."""
    return np.asarray(j).reshape(len(j), -1).sum(axis = 1)


def metrics(t_ref, y_ref, t_cmp, y_cmp):
    """max and rms of |y_ref - y_cmp| over the overlapping time window."""
    m = (t_cmp >= t_ref.min()) & (t_cmp <= t_ref.max())
    d = np.abs(np.interp(t_cmp[m], t_ref, y_ref) - y_cmp[m])
    return d.max(), np.sqrt((d ** 2).mean())


print('\nLeft current vs scipy_ode (reference)')
print('%-24s %12s %12s' % ('pair', 'max|dJ|', 'rms|dJ|'))
for name, t, j in (('scipy_ode - zand',   t_z, total(jz[0])),
                   ('scipy_ode - nozand', t_n, total(jn[0]))):
    mx, rms = metrics(sp_t, sp_jl, t, j)
    print('%-24s %12.4g %12.4g' % (name, mx, rms))

mx, rms = metrics(t_z, total(jz[0]), t_n, total(jn[0]))
print('%-24s %12.4g %12.4g' % ('zand - nozand', mx, rms))
print()
print('\nParticle number at the end (scipy_ode only keeps k=0)')
print('scipy_ode', np.round(sp_N[-1], 4))
print('zand     ', np.round(N_z[-1].real, 4))
print('nozand   ', np.round(N_n[-1].real, 4), '\n')

fig, ax = plt.subplots(1, 2, figsize = (12, 4.5))

# Left currents solid, right currents dashed.
ax[0].plot(sp_t, sp_jl, color = 'k',  label = 'scipy_ode (left)')
ax[0].plot(t_z, total(jz[0]),  color = 'C0', label = 'zand (left)')
ax[0].plot(t_z, total(jz[1]),  color = 'C0', linestyle = 'dashed', label = 'zand (right)')
ax[0].plot(t_n, total(jn[0]),  color = 'C3', label = 'nozand (left)')
ax[0].plot(t_n, total(jn[1]),  color = 'C3', linestyle = 'dashed', label = 'nozand (right)')
ax[0].set_xlabel('time')
ax[0].set_ylabel('Current')
ax[0].legend(fontsize = 8)

for k in range(N_z.shape[1]):
    ax[1].plot(t_Nz, N_z[:, k].real, color = 'C0', label = 'zand, k=%d' % k)
for k in range(N_n.shape[1]):
    ax[1].plot(t_Nn, N_n[:, k].real, color = 'C3', linestyle = 'dashed',
               label = 'nozand, k=%d' % k)
ax[1].plot(sp_t, sp_N, color = 'k', linestyle = 'dotted', label = 'scipy_ode, k=0')
ax[1].set_xlabel('time')
ax[1].set_ylabel(r'$N_k$')
ax[1].legend(fontsize = 8)

plt.tight_layout()
plt.savefig('Rib_compare.png', dpi = 300)
plt.close()
