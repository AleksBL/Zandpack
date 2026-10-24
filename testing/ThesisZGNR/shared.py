"""Drive and dH for the ThesisZGNR test -- orthogonal (Lowdin) basis.

Adapted from examples/ThesisExample/Bias.py: a bias box pulse on the leads and
a mean-field term that raises/lowers two mid-ribbon sites in proportion to
their charge.  The density matrix enters `dH`, so the equations of motion are
non-linear.

The drive term is linear in the bias, so L @ diag(Rmp*V) @ L = V*M with
M = L @ diag(Rmp) @ L precomputed once -- no matrix products at call time.

Both functions are numba-jitted because the shortlib right-hand side calls
them from compiled code (the box-pulse math is inlined -- the package's
box_pulse is not jitted).  This is a real SIESTA device (S != I), so the
nozand twin (Bias_NO.py) must transform BOTH the argument and the result.
"""

import numpy as np
from numba import njit
from sisl import get_sile

# ---- tuning block ---------------------------------------------------------
name = 'TDZGNR'
Rep    = 2.0          # mean-field coupling (as in ThesisExample)
tp, ts = 10.0, 1.5    # box pulse: on at t ~ tp, smeared over ts
K1, K2 = 23, 24       # the two mean-field sites (pz indices, mid-ribbon)
# ---------------------------------------------------------------------------

pref = name + '/Arrays/'
L  = np.load(pref + 'S^(-0.5).npy')
p  = np.load(pref + 'pivot.npy')
Hs = get_sile('Dev/siesta.TSHS').read_hamiltonian()
r  = np.load(pref + 'Positions.npy')
r  = r[[Hs.o2a(i) for i in p]]
Rmp = (r[:, 0] - np.average(r[:, 0]))
Rmp /= (r[:, 0].max() - r[:, 0].min())

# V-independent part of the drive, in the orthogonal basis
M = (L @ np.diag(Rmp) @ L).astype(np.complex128)


@njit
def bias(t, a):
    """box_pulse(t, tp, ts, 1), inlined so it can be jitted."""
    box = (np.tanh(t / ts) - np.tanh((t - tp) / ts)) / 2
    return np.sign(a - .5) * (1 - box) * .5


@njit
def dH(t, sigma):
    """Linear drive from the bias pulse plus the charge mean-field on K1, K2."""
    V  = (bias(t, 1) - bias(t, 0)) * 0.5
    dh = V * M
    dh[0, K1, K1] += Rep * sigma[0, K1, K1]
    dh[0, K2, K2] += Rep * sigma[0, K2, K2]
    return dh


@njit
def dissipator(t, sig):
    return 0.0
