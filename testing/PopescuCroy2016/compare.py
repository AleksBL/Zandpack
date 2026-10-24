"""Compare the scipy_ode (and AdaptiveRK4) propagators against the zand and
nozand MPI codes.

scipy_ode is the reference for the printed differences; the figure shows the
left/right currents.  The metrics are printed so the test can be read without
looking at the plot.
"""

import numpy as np
from Zandpack.plot import plt, J

R1    = np.load("croy2016_results.npz")
JL    = R1["rk4_jl"].ravel();     JR    = R1["rk4_jr"].ravel()
T     = R1["rk4_t"]
sp_jl = R1["scipy_jl"].ravel();  sp_t  = R1["scipy_t"]

t_n, j_n = J(["TDT_Croy2016_save_nozand"])
t_z, j_z = J(["TDT_Croy2016_save_zand"])


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
for name, t, j in (('scipy_ode - AdaptiveRK4', T,   JL),
                   ('scipy_ode - zand',        t_z, total(j_z[0])),
                   ('scipy_ode - nozand',      t_n, total(j_n[0]))):
    mx, rms = metrics(sp_t, sp_jl, t, j)
    print('%-24s %12.4g %12.4g' % (name, mx, rms))

# the two MPI codes should agree with each other
mx, rms = metrics(t_z, total(j_z[0]), t_n, total(j_n[0]))
print('%-24s %12.4g %12.4g' % ('zand - nozand', mx, rms))
print()

plt.plot(T, JL, label = 'AdaptiveRK4 (left)')
plt.plot(T, JR, label = 'AdaptiveRK4 (right)')
plt.plot(t_n, total(j_n[0]), linestyle = "dashed",  label = 'nozand (left)')
plt.plot(t_n, total(j_n[1]), linestyle = "dotted",  label = 'nozand (right)')
plt.plot(t_z, total(j_z[0]), linestyle = "dashdot", label = 'zand (left)')
plt.scatter(sp_t, sp_jl, s = 5, label = 'scipy_ode (left)')
plt.xlabel('time')
plt.ylabel('Current')
plt.legend(fontsize = 8)
from time import strftime   # dated copy of every run's figure
stamp = strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig("Croy2016_compare.png", dpi = 300)
plt.savefig('Croy2016_compare_%s.png' % stamp, dpi = 300)
plt.close()
