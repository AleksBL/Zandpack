import os
ATOL = float(os.environ.get('ZP_ATOL', 1e-8))   # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', 1e-8))
T0 = float(os.environ.get('ZP_T0', 0))   # fast mode, see run_zandpack_tests --fast
T1 = float(os.environ.get('ZP_T1', 20))
# scipy_ode runs one solve_ivp per eval point and its memory grows with that
# count, not only with the window (measured on this device at t = 0..2:
# N = 50 -> >12 GB, N = 10 -> 3.5 GB; SR's ODE state is 25 MB to begin with, see
# handoff.md).  So the default here is deliberately small.
# Output times. scipy_ode's memory and its interpolated metric both grow with
# the number of eval points, and this device's current varies faster than a
# uniform N = 20 grid resolves -- so the default is a sparse list of
# well-chosen times, compared pointwise (max|dJ| only, scatter crosses).
# ZP_TLIST="0.5,2,10" overrides the list; ZP_NEVAL (e.g. fast mode) restores
# a uniform grid of N points.
if 'ZP_TLIST' in os.environ:
    TL = [float(x) for x in os.environ['ZP_TLIST'].split(',')]
elif 'ZP_NEVAL' in os.environ:
    N  = int(os.environ['ZP_NEVAL'])
    TL = None
else:
    TL = [t for t in (0.1, 0.5, 1, 2, 5, 10, 15, 20) if t <= T1]
    if not TL:
        TL = [T1]

"""Compare scipy_ode against the zand and nozand MPI codes on the STM device.

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
from Zandpack.plot import plt, J
from Zandpack.plot import N as occupation      # keep `N` free for the sample count
from Zandpack.Help import TDHelper
from shared import dH, delta

C = load(open('Fit.xz','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(zero_tol = 1e-8)

# Start from the same state the MPI codes read; this also picks up any
# refinement done by SCF/psinought, which rewrites these arrays.
from Zandpack.Loader import flexload
A       = 'STM2/Arrays/'
C.sigma = flexload(A + 'DM_Ortho.npy')
C.xi    = flexload(A + 'xi.npz')
C.Ixi   = flexload(A + 'Ixi.npz')
C.get_propagation_quantities()      # rebuild psi/omega from xi/Ixi
C.Check_input_to_ODE()

# shortlib lives at the TestRUN root, whatever the depth of this file
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != '/' and not _os.path.isdir(_os.path.join(_root, 'shortlib')):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from shortlib.driver import propagate          # ZP_RHS / ZP_CHUNK

t_eval = np.array(TL) + 1e-5 if TL is not None else np.linspace(T0, T1, N) + 1e-5
t0, dm, jl = propagate(C, dH, delta, 0.0, t_eval, ATOL, RTOL)

sparse = TL is not None   # sparse output grid: compare pointwise, max only

sp_t  = t0
sp_jl = jl.sum(axis = 1)
sp_N  = np.trace(dm, axis1 = -2, axis2 = -1).real

# save like the other tests, so the metrics can be re-analysed offline
np.savez('scipy_ode_results.npz', t = sp_t, jl = jl, N = sp_N)

t_z, jz = J(['STM2_save_zand'])
t_n, jn = J(['STM2_save_nozand'])

Hlp       = TDHelper('STM2')
t_Nz, N_z = occupation(['STM2_save_zand'])
t_Nn, N_n = occupation(['STM2_save_nozand'], S = Hlp.S, insert_tril = True)


def total(j):
    """zand/nozand report one current column per k-point; sum for the total."""
    return np.asarray(j).reshape(len(j), -1).sum(axis = 1)


def metrics(t_ref, y_ref, t_cmp, y_cmp):
    """max and rms of |y_ref - y_cmp| over the overlapping time window."""
    m = (t_cmp >= t_ref.min()) & (t_cmp <= t_ref.max())
    d = np.abs(np.interp(t_cmp[m], t_ref, y_ref) - y_cmp[m])
    return d.max(), np.sqrt((d ** 2).mean())


def metrics_at_points(t_dense, y_dense, t_pts, y_pts):
    """max |dJ| at the sample points themselves: the dense curve is sampled at
    the sparse scipy_ode times.  Never interpolate a coarse curve onto a fine
    grid -- that manufactures differences the methods do not have."""
    m = (t_pts >= t_dense.min()) & (t_pts <= t_dense.max())
    d = np.abs(y_pts[m] - np.interp(t_pts[m], t_dense, y_dense))
    return d.max()


if sparse:
    print('\nLeft current vs scipy_ode at its sample points (coarse grid: max|dJ| only)')
    print('%-24s %12s' % ('pair', 'max|dJ|'))
    for name, t, j in (('scipy_ode - zand',   t_z, total(jz[0])),
                       ('scipy_ode - nozand', t_n, total(jn[0]))):
        print('%-24s %12.4g' % (name, metrics_at_points(t, j, sp_t, sp_jl)))
    print('%-24s %12.4g' % ('zand - nozand', metrics(t_z, total(jz[0]), t_n, total(jn[0]))[0]))
else:
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

# Left currents solid, right currents dashed.  A sparse scipy_ode grid is
# drawn as scatter crosses -- a line through few points would suggest
# structure that was never computed.
if sparse:
    ax[0].scatter(sp_t, sp_jl, marker = 'x', s = 30, linewidths = 1.0,
                  color = 'k', label = 'scipy_ode (left, sample points)')
    ax[1].scatter(sp_t, sp_N, marker = 'x', s = 30, linewidths = 1.0,
                  color = 'k', label = 'scipy_ode, k=0 (sample points)')
else:
    ax[0].plot(sp_t, sp_jl, color = 'k',  label = 'scipy_ode (left)')
    ax[1].plot(sp_t, sp_N, color = 'k', linestyle = 'dotted', label = 'scipy_ode, k=0')
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
ax[1].set_xlabel('time')
ax[1].set_ylabel(r'$N_k$')
ax[1].legend(fontsize = 8)

plt.tight_layout()
from time import strftime   # dated copy of every run's figure
stamp = strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig('SR_compare.png', dpi = 300)
plt.savefig('SR_compare_%s.png' % stamp, dpi = 300)
plt.close()
