#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where does shortlib's RHS spend its time?  (assessment helper, not a test)

Builds three variants of the same numba function by doctoring the *index maps*
(still the same arithmetic, just with blocks switched off):

    full        everything
    no_omega    om_ptr zeroed      -> D_omega and the omega-sum term do nothing
    sig_only    + psi_mask cleared -> only the [H,sigma] and Pi part runs

so `full - no_omega` is the omega cost and `no_omega - sig_only` the auxiliary
mode cost.  Also times the package's make_f for reference.

Run it for **one device per process** (`python profile_rhs.py SR`).  Loading
several devices and compiling all their variants in one process crashed this
environment with a heap corruption (`corrupted size vs. prev_size`), so the
default is a single device.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from verify_rhs import load_device                    # noqa: E402
from shortlib.rhs import arrays_from_td, make_f_fast  # noqa: E402
from Zandpack.td_constants import hbar                # noqa: E402

DEVICES = {'SR': ('SR', 'Fit.xz', 'STM2'),
           'scipy_ode_vs_zand': ('scipy_ode_vs_zand', 'Graw.xz', 'SRC'),
           'synth_dm': ('synth_dm', 'synth_dm.xz', 'synth_dm')}


def variants(A):
    out = {'full': A}
    A2 = dict(A)
    A2['maps'] = dict(A['maps'])
    A2['maps']['om_ptr'] = np.zeros_like(A['maps']['om_ptr'])
    out['no_omega'] = A2
    A3 = dict(A2)
    A3['maps'] = dict(A2['maps'])
    A3['maps']['psi_mask'] = np.zeros_like(A['maps']['psi_mask'])
    out['sig_only'] = A3
    return out


def timeit(f, args, n):
    f(*args)                                   # compile
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        f(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    for name in sys.argv[1:] or ['SR']:
        d, pkl, arr = DEVICES[name]
        C, dH, delta = load_device(d, pkl, arr)
        A = arrays_from_td(C)
        nk, no, nl, Ntot = A['nk'], A['no'], A['nl'], A['Ntot']
        n_mode = nl * Ntot * no
        n_om = A['maps']['om_pack'].shape[0]
        args = (1.7, C.sigma, C.Psi_vec, C.omegas, dH, delta, True)

        print('=' * 70)
        print('%s: nk=%d no=%d n_mode=%d  omega entries=%d (%.1f %% of the dense '
              'map)  psi-active modes=%d/%d'
              % (name, nk, no, n_mode, n_om, 100.0 * n_om / n_mode ** 2,
                 int(A['maps']['psi_mask'].sum()), n_mode))

        times = {}
        for label, AA in variants(A).items():
            f = make_f_fast(AA, hbar=float(hbar))
            times[label] = timeit(f, args, 5)
        f_ref = C.make_f()
        t_ref = timeit(f_ref, args, 5)

        t_all, t_no, t_sig = times['full'], times['no_omega'], times['sig_only']
        print('  %-22s %9.2f ms' % ('make_f (reference)', 1e3 * t_ref))
        print('  %-22s %9.2f ms   (%.2fx vs make_f)' % ('shortlib full', 1e3 * t_all, t_ref / t_all))
        print('  %-22s %9.2f ms   -> omega block %.2f ms (%.0f %%)'
              % ('  ... without omega', 1e3 * t_no, 1e3 * (t_all - t_no),
                 100.0 * (t_all - t_no) / t_all))
        print('  %-22s %9.2f ms   -> psi block   %.2f ms (%.0f %%)'
              % ('  ... sigma only', 1e3 * t_sig, 1e3 * (t_no - t_sig),
                 100.0 * (t_no - t_sig) / t_all))
        print('  %-22s %9.2f ms   (%.0f %%)'
              % ('  ... [H,sig]+Pi', 1e3 * t_sig, 100.0 * t_sig / t_all))
        print('  per-omega-entry cost: %.1f ns'
              % (1e9 * (t_all - t_no) / max(n_om * nk, 1)))
        print()


if __name__ == '__main__':
    main()
