"""Compare scipy_ode against the zand and nozand MPI codes on the synthetic
device, and print the difference metrics.

Left  panel: the current, read through Zandpack.plot.J for the MPI runs.
Right panel: Tr(DM) per k-point, read through Zandpack.plot.N. nozand works in
             the nonorthogonal basis, so it is given the overlap S to report
             the same particle number as the orthogonal runs.
"""

import numpy as np
import sys
from Zandpack.plot import plt, J, N
from Zandpack.Help import TDHelper

# "saw" picks the sawtooth run (run_scipy_ode.py saw + Bias_saw.py)
tag = '_saw' if (len(sys.argv) > 1 and sys.argv[1] == 'saw') else ''

R     = np.load('scipy_ode_results%s.npz' % tag)
sp_t  = R['t']
sp_jl = R['jl'].sum(axis = 1)
sp_N  = R['N']

t_z, jz = J(['synth_save_zand%s'   % tag])
t_n, jn = J(['synth_save_nozand%s' % tag])

Hlp       = TDHelper('synth')
t_Nz, N_z = N(['synth_save_zand%s'   % tag])
t_Nn, N_n = N(['synth_save_nozand%s' % tag], S = Hlp.S, insert_tril = True)


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
from time import strftime   # dated copy of every run's figure
stamp = strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig('synth_nozand_vs_scipy_ode%s.png' % tag, dpi = 300)
plt.savefig('synth_nozand_vs_scipy_ode%s_%s.png' % (tag, stamp), dpi = 300)
plt.close()
