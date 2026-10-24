"""Compare the three propagation methods on the Honeychurch2018 device.

`scipy_ode` is the reference (it is on the Python side of every other test too);
`AdaptiveRK4` -- the paper's own Python propagator -- and the `zand`/`nozand`
MPI codes are measured against it, and the two MPI codes against each other.

This is the test whose AdaptiveRK4 differed most from the MPI codes, so all
three pairs are printed: the point is to see whether scipy_ode agrees with the
MPI codes better than AdaptiveRK4 does. See handoff.md.

Left  panel: the current, read through Zandpack.plot.J for the MPI runs.
Right panel: Tr(DM) per k-point, read through Zandpack.plot.N. nozand works in
             the nonorthogonal basis, so it is given the overlap S to report the
             same particle number as the orthogonal runs.
"""

import numpy as np
from Zandpack.plot import plt, J, N
from Zandpack.Help import TDHelper

R     = np.load('honeychurch2018_results.npz')
sp_t  = R['scipy_t']
sp_jl = np.asarray(R['scipy_jl']).reshape(len(sp_t), -1).sum(axis = 1)
sp_N  = R['scipy_N']

TL = R['rk4_t'].ravel();  JL = R['rk4_jl'].ravel();  JR = R['rk4_jr'].ravel()

t_n, j_n = J(['TDT_hc2018_save_nozand'])
t_z, j_z = J(['TDT_hc2018_save_zand'])

Hlp       = TDHelper('TDT')
t_Nz, N_z = N(['TDT_hc2018_save_zand'])
t_Nn, N_n = N(['TDT_hc2018_save_nozand'], S = Hlp.S, insert_tril = True)


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
pairs = (('scipy_ode - AdaptiveRK4', sp_t, sp_jl,           TL, JL),
         ('scipy_ode - zand',        sp_t, sp_jl,           t_z, total(j_z[0])),
         ('scipy_ode - nozand',      sp_t, sp_jl,           t_n, total(j_n[0])),
         ('AdaptiveRK4 - zand',      TL,   JL,              t_z, total(j_z[0])),
         ('AdaptiveRK4 - nozand',    TL,   JL,              t_n, total(j_n[0])),
         ('zand - nozand',           t_z,  total(j_z[0]),   t_n, total(j_n[0])))
for name, tr, yr, tc, yc in pairs:
    mx, rms = metrics(tr, yr, tc, yc)
    print('%-24s %12.4g %12.4g' % (name, mx, rms))
print()
print('\nParticle number at the end (scipy_ode / AdaptiveRK4 keep k=0 only)')
print('scipy_ode', np.round(sp_N[-1], 4))
print('zand     ', np.round(N_z[-1].real, 4))
print('nozand   ', np.round(N_n[-1].real, 4), '\n')

fig, ax = plt.subplots(1, 2, figsize = (12, 4.5))

ax[0].plot(sp_t, sp_jl, color = 'k',  label = 'scipy_ode (left)')
ax[0].plot(TL,   JL,    color = 'C2', label = 'AdaptiveRK4 (left)')
ax[0].plot(TL,   JR,    color = 'C2', linestyle = 'dashdot', label = 'AdaptiveRK4 (right)')
ax[0].plot(t_z, total(j_z[0]), color = 'C0', label = 'zand (left)')
ax[0].plot(t_z, total(j_z[1]), color = 'C0', linestyle = 'dashed', label = 'zand (right)')
ax[0].plot(t_n, total(j_n[0]), color = 'C3', label = 'nozand (left)')
ax[0].plot(t_n, total(j_n[1]), color = 'C3', linestyle = 'dashed', label = 'nozand (right)')
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
plt.savefig("Honeychurch2018_compare.png", dpi = 300)
plt.savefig('Honeychurch2018_compare_%s.png' % stamp, dpi = 300)
plt.close()
