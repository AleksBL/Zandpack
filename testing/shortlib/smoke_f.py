#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test: is shortlib's f equivalent to the package's ``make_f``, and faster?

Short on purpose.  For the thorough version -- three devices, many perturbed
states, Hermiticity and packing invariants, ``hbar != 1``, both back ends -- use
``verify_rhs.py``.

    python smoke_f.py [device]        # default: synth_dm
"""

import sys
import time

import numpy as np

from verify_rhs import load_device, worst, DEVICES
from shortlib.rhs import arrays_from_td, make_f_fast
from Zandpack.td_constants import hbar

DEV = sys.argv[1] if len(sys.argv) > 1 else 'synth_dm'
d, pkl, arr = next(x for x in DEVICES if x[0] == DEV)

C, dH, delta = load_device(d, pkl, arr)
A = arrays_from_td(C)
f_ref = C.make_f()
variants = [('dense_omega', make_f_fast(A, hbar=float(hbar), dense_omega=True)),
            ('packed',      make_f_fast(A, hbar=float(hbar), dense_omega=False))]

sig, psi, omg = C.sigma, C.Psi_vec, C.omegas
rng = np.random.default_rng(0)
sig_r = np.ascontiguousarray(sig + 0.05 * (rng.standard_normal(sig.shape)
                                           + 1j * rng.standard_normal(sig.shape)))
psi_r = np.ascontiguousarray(psi + 0.2 * (rng.standard_normal(psi.shape)
                                          + 1j * rng.standard_normal(psi.shape)))
omg_r = np.ascontiguousarray(omg + 0.3 * (rng.standard_normal(omg.shape)
                                          + 1j * rng.standard_normal(omg.shape)))

print('%s: nk=%d no=%d nlead=%d Ntot=%d  n_mode=%d  packed omega nz=%d  hbar=%r'
      % (d, A['nk'], A['no'], A['nl'], A['Ntot'], A['nl'] * A['Ntot'] * A['no'],
         A['nz'], float(hbar)))
print()
print('%-13s %-9s %-11s %-11s %-11s' % ('state', 'backend', 'dSigma', 'dPsi', 'dOmega'))
worst_rel = {name: 0.0 for name, _ in variants}
for name, s, p, o in (('real', sig, psi, omg), ('random', sig_r, psi_r, omg_r)):
    R = f_ref(1.3, s, p, o, dH, delta, True)
    for vname, fun in variants:
        G = fun(1.3, s, p, o, dH, delta, True)
        e = [worst(R[i], G[i]) for i in range(3)]
        worst_rel[vname] = max(worst_rel[vname], max(r for _, r in e))
        print('%-13s %-9s %-11.2e %-11.2e %-11.2e   abs' % (name, vname, *[a for a, _ in e]))
        print('%-13s %-9s %-11.2e %-11.2e %-11.2e   rel' % ('', '', *[r for _, r in e]))


def timeit(fun, n=3):
    fun(1.0, sig, psi, omg, dH, delta, True)          # compile
    t0 = time.perf_counter()
    for _ in range(n):
        fun(1.0, sig, psi, omg, dH, delta, True)
    return (time.perf_counter() - t0) / n


# the shipped default must be the packed back end
G_def = make_f_fast(A, hbar=float(hbar))(1.3, sig, psi, omg, dH, delta, True)
G_pk = variants[1][1](1.3, sig, psi, omg, dH, delta, True)
d_def = max(np.abs(G_def[i] - G_pk[i]).max() for i in range(3))
print()
print('default back end == packed (dense_omega=False): %s'
      % ('yes' if d_def == 0.0 else 'NO, differs by %.2e' % d_def))

print()
print('per RHS call:')
vals = [('make_f (reference)', timeit(f_ref))]
for vname, fun in variants:
    vals.append((vname, timeit(fun)))
base = vals[0][1]
for label, tt in vals:
    print('  %-18s %8.3f ms   %5.2fx' % (label, 1e3 * tt, base / tt))
ok = all(v < 1e-10 for v in worst_rel.values())
print()
print('EQUIVALENT to make_f (worst relative deviation %s)'
      % ', '.join('%s %.1e' % (k, v) for k, v in worst_rel.items())
      if ok else 'MISMATCH')
sys.exit(0 if ok else 1)
