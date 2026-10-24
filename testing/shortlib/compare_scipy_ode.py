#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the piecewise driver reproduce scipy_ode, and does shortlib's f reproduce
the tests' results?

Four variants, each run in **its own process** so the peak RSS is meaningful:

    ref         package scipy_ode            + package make_f   (the baseline)
    ref_new     package scipy_ode            + shortlib f
    pieces      shortlib.solve_pieces        + package make_f
    pieces_new  shortlib.solve_pieces        + shortlib f

Usage
    python compare_scipy_ode.py run <device> <variant> <out.npz> [--chunk K] [--t1 T] [--n N]
    python compare_scipy_ode.py compare <out1.npz> <out2.npz> ...

``chunk = 1`` gives the package's call pattern exactly (one solver call per eval
point), so it is the control.  ``compare`` reports, against ``ref``, the largest
deviation of the left current and of the k=0 density matrix, together with each
run's wall time and peak RSS.
"""

import os
import sys
import time
import resource

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from verify_rhs import load_device                    # noqa: E402
from shortlib.rhs import arrays_from_td, make_f_fast  # noqa: E402
from shortlib.piecewise import solve_pieces           # noqa: E402
from Zandpack.decoupled_functions import scipy_ode    # noqa: E402
from Zandpack.td_constants import hbar                # noqa: E402

# device -> (dir, pickle, Arrays/, t0, t1, N, atol, rtol).  These are the
# defaults the tests themselves use.
DEVICES = {
    'synth_1':            ('synth_1',            'synth.xz',   'synth',       0.0, 20.0, 200, 1e-8, 1e-8),
    'synth_dm':           ('synth_dm',           'synth_dm.xz', 'synth_dm',   0.0, 20.0, 200, 1e-8, 1e-8),
    'SR':                 ('SR',                 'Fit.xz',      'STM2',       0.0, 20.0,  20, 1e-8, 1e-8),
    'scipy_ode_vs_zand':  ('scipy_ode_vs_zand',  'Graw.xz',     'SRC',        0.0, 20.0, 200, 1e-8, 1e-8),
    'small_bound':        (os.path.join('examples', 'small_bound'), 'Testfit.xz', 'small_bound',
                                                                                  0.0, 20.0, 200, 1e-7, 1e-5),
}


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def do_run(device, variant, out, chunk=1, t1=None, n=None):
    d, pkl, arr, t0, t1_def, n_def, atol, rtol = DEVICES[device]
    t1 = t1_def if t1 is None else t1
    n = n_def if n is None else n
    C, dH, delta = load_device(d, pkl, arr)
    A = arrays_from_td(C)
    f_pkg = C.make_f()
    f_new = make_f_fast(A, hbar=float(hbar))          # default back end
    f = f_new if variant.endswith('_new') else f_pkg

    t_eval = np.linspace(t0, t1, n) + 1e-5
    kw = dict(dH_given=True, atol=atol, rtol=rtol)
    if variant.startswith('ref'):
        kw.pop('dH_given')
        call = lambda: scipy_ode(f, C.sigma, C.Psi_vec, C.omegas, t0, t_eval,
                                 dH, delta, C.Ixi, **kw)
    else:
        call = lambda: solve_pieces(f, C.sigma, C.Psi_vec, C.omegas, t0, t_eval,
                                    dH, delta, C.Ixi, chunk=chunk, atol=atol, rtol=rtol)

    t_start = time.perf_counter()
    t, rho, cl = call()
    secs = time.perf_counter() - t_start
    np.savez_compressed(out, t=t, rho=np.asarray(rho), cl=np.asarray(cl),
                        secs=secs, peak_mb=peak_rss_mb(), chunk=chunk, variant=variant)
    print('%-11s %-11s chunk=%-4d  %7.1f s   peak RSS %7.0f MB   -> %s'
          % (device, variant, chunk, secs, peak_rss_mb(), out))


def do_compare(ref_path, others):
    r = np.load(ref_path)
    print('reference: %s   (%s, chunk=%d, %.1f s, peak %.0f MB)'
          % (os.path.basename(ref_path), r['variant'], r['chunk'], r['secs'], r['peak_mb']))
    print()
    print('%-38s %12s %12s %9s %10s' % ('variant', 'max|dJ_l|', 'max|dN|', 'secs', 'peak MB'))
    print('%-38s %12.3e %12.3e %9.1f %10.0f'
          % ('ref (baseline)', 0.0, 0.0, r['secs'], r['peak_mb']))
    for path in others:
        o = np.load(path)
        assert np.allclose(o['t'], r['t']), 'time grids differ'
        dJ = np.abs(np.asarray(o['cl']).reshape(len(o['t']), -1).sum(1)
                    - np.asarray(r['cl']).reshape(len(r['t']), -1).sum(1)).max()
        dN = np.abs(np.trace(np.asarray(o['rho']), axis1=-2, axis2=-1).real
                    - np.trace(np.asarray(r['rho']), axis1=-2, axis2=-1).real).max()
        label = '%s (chunk=%d)' % (o['variant'], o['chunk'])
        print('%-38s %12.3e %12.3e %9.1f %10.0f'
              % (label, dJ, dN, o['secs'], o['peak_mb']))
    print()
    print('currents: max|dJ_l| against ref;  density matrix: max|dN| of Tr(rho)')


def main():
    mode = sys.argv[1]
    if mode == 'run':
        device, variant, out = sys.argv[2], sys.argv[3], sys.argv[4]
        opts = dict(zip(sys.argv[5::2], sys.argv[6::2]))
        do_run(device, variant, out,
               chunk=int(opts.get('--chunk', 1)),
               t1=float(opts['--t1']) if '--t1' in opts else None,
               n=int(opts['--n']) if '--n' in opts else None)
    elif mode == 'compare':
        do_compare(sys.argv[2], sys.argv[3:])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
