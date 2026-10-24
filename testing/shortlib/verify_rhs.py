#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thorough verification of ``shortlib.rhs`` against ``TD_Transport.make_f``.

For every device listed in DEVICES below it

  1.  builds the reference and *both* shortlib back ends from the same arrays,
  2.  checks invariants that hold independently of the reference:
        * ``D_sig`` is Hermitian for a Hermitian ``sigma`` (the EOM preserves
          hermiticity, so a wrong ``[H,sig]`` or ``Pi+Pi^dag`` shows up here);
        * both implementations satisfy it too (a cross-check on the
          transcription of the equations, not just on agreement),
  3.  compares ``D_sig``, ``D_psi`` and ``D_omega`` for
        * the real propagation state at three different times,
        * random perturbations of each block, so the sparse/zero entries of
          ``psi``/``omega`` are exercised rather than skipped,
        * a second, time-dependent ``dH``,
  4.  reports the worst absolute and relative deviation for each back end, the
      fill fraction of the index map, and a timing of all three.

Run it from here::

    python verify_rhs.py [reps]           # default reps=2
    ZANDPACK_HBAR=2.5 python verify_rhs.py    # also checks the hbar placement

Exit status 0 means every comparison passed.
"""

import os
import sys
import time
import lzma
from pickle import load

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)              # TestRUN/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from shortlib.rhs import arrays_from_td, make_f_fast      # noqa: E402
from Zandpack.Loader import flexload                       # noqa: E402
from Zandpack.td_constants import hbar                     # noqa: E402

# (directory, pickle, Arrays/ name).  Chosen to cover the shape space and both
# regimes of the index map: 2 orbitals/1 k; 12 orbitals/1 k with a large Ntot
# (fill 36 %, where the dense back end wins); 4 orbitals/10 k with a very sparse
# map (fill 7.5 %, where it loses).
DEVICES = [
    ('synth_dm',          'synth_dm.xz', 'synth_dm'),
    ('SR',                'Fit.xz',      'STM2'),
    ('scipy_ode_vs_zand', 'Graw.xz',     'SRC'),
]

TOL_REL = 1e-10        # both sides are fastmath; expect ~1e-15


@njit
def dH_alt(t, sigma):
    """A second, time-dependent dH, to make sure `t` is threaded through."""
    out = np.zeros(sigma.shape, dtype=np.complex128)
    out[0, 0, 0] = 0.37 * np.sin(t / 5.0)
    if sigma.shape[1] > 1:
        out[0, -1, -1] = -0.21 * np.cos(t / 3.0)
    return out


def load_device(d, pkl, arrname):
    """Load a TD_Transport and put it in the state its own driver leaves it."""
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, d))
    try:
        C = load(lzma.open(pkl, 'rb'))
        C.diagonalise()
        C.get_propagation_quantities()
        C.get_dense_matrices(zero_tol=1e-8)
        A = arrname + '/Arrays/'
        C.sigma = flexload(A + 'DM_Ortho.npy')
        C.xi    = flexload(A + 'xi.npz')
        C.Ixi   = flexload(A + 'Ixi.npz')
        C.get_propagation_quantities()
        sys.path.insert(0, os.getcwd())
        from shared import dH, delta                       # noqa: E402
    finally:
        os.chdir(cwd)
    return C, dH, delta


def worst(ref, got):
    """(max abs deviation, deviation relative to the scale of ref)."""
    ref, got = np.asarray(ref), np.asarray(got)
    assert ref.shape == got.shape, (ref.shape, got.shape)
    d = np.abs(ref - got)
    scale = np.abs(ref).max()
    return d.max(), (d.max() / scale if scale > 0 else 0.0)


def herm_err(M):
    return np.abs(np.asarray(M) - np.conj(np.swapaxes(M, -1, -2))).max()


def check_device(d, pkl, arrname, reps):
    print('=' * 78)
    print('%s   (%s / %s)' % (d, pkl, arrname))
    print('=' * 78)
    C, dH, delta = load_device(d, pkl, arrname)

    sig0, psi0, omg0 = C.sigma, C.Psi_vec, C.omegas
    A = arrays_from_td(C)
    f_ref = C.make_f()
    variants = [('dense_omega', make_f_fast(A, hbar=float(hbar), dense_omega=True)),
                ('packed',      make_f_fast(A, hbar=float(hbar), dense_omega=False))]
    nk, no, nl, Ntot = A['nk'], A['no'], A['nl'], A['Ntot']
    mp = A['maps']
    n_mode = nl * Ntot * no
    n_om = mp['om_pack'].shape[0]
    print('nk=%d no=%d nlead=%d Ntot=%d   n_mode=%d   packed omega nz=%d   hbar=%r'
          % (nk, no, nl, Ntot, n_mode, A['nz'], float(hbar)))
    print('index map: %d of %d (row,src) pairs stored (%.1f %%), dense map would '
          'be %.0f MB of int64'
          % (n_om, n_mode * n_mode, 100.0 * n_om / (n_mode * n_mode),
             n_mode * n_mode * 8 / 1e6))
    ok = True

    def states():
        yield 'real state', sig0, psi0, omg0
        for rep in range(reps):
            rng = np.random.default_rng(1000 + rep)
            sig = sig0 + 0.05 * (rng.standard_normal(sig0.shape)
                                 + 1j * rng.standard_normal(sig0.shape))
            sig = 0.5 * (sig + np.conj(np.swapaxes(sig, -1, -2)))   # Hermitian
            psi = psi0 + 0.2 * (rng.standard_normal(psi0.shape)
                                + 1j * rng.standard_normal(psi0.shape))
            omg = omg0 + 0.3 * (rng.standard_normal(omg0.shape)
                                + 1j * rng.standard_normal(omg0.shape))
            yield ('random #%d' % (rep + 1),
                   np.ascontiguousarray(sig, dtype=np.complex128),
                   np.ascontiguousarray(psi, dtype=np.complex128),
                   np.ascontiguousarray(omg, dtype=np.complex128))
        # a sparser state, to weight the pieces differently
        rng = np.random.default_rng(777)
        psi = (rng.random(psi0.shape) < 0.2) * (rng.standard_normal(psi0.shape)
                                                + 1j * rng.standard_normal(psi0.shape))
        omg = (rng.random(omg0.shape) < 0.2) * (rng.standard_normal(omg0.shape)
                                                + 1j * rng.standard_normal(omg0.shape))
        yield 'sparse state', sig0, np.ascontiguousarray(psi), np.ascontiguousarray(omg)

    worst_rel = {name: 0.0 for name, _ in variants}
    worst_abs = {name: 0.0 for name, _ in variants}
    for name, sig, psi, omg in states():
        for tag, t, dh in (('dH=shared', 1.7, dH), ('dH=alt', -3.25, dH_alt)):
            R = f_ref(t, sig, psi, omg, dh, delta, dH_given=True)
            if herm_err(R[0]) > 1e-10:
                print('FAIL  reference D_sig is not Hermitian (%.2e)' % herm_err(R[0]))
                ok = False
            for vname, fun in variants:
                G = fun(t, sig, psi, omg, dh, delta, dH_given=True)
                e = [worst(R[i], G[i]) for i in range(3)]
                worst_abs[vname] = max(worst_abs[vname], max(a for a, _ in e))
                worst_rel[vname] = max(worst_rel[vname], max(r for _, r in e))
                if herm_err(G[0]) > 1e-10:
                    print('FAIL  %s D_sig is not Hermitian (%.2e)'
                          % (vname, herm_err(G[0])))
                    ok = False
                if max(r for _, r in e) > TOL_REL:
                    print('FAIL  %s on %-12s %-10s  dSigma %.2e dPsi %.2e dOmega %.2e'
                          % (vname, name, tag, e[0][1], e[1][1], e[2][1]))
                    ok = False
        print('%-13s compared (dSigma/dPsi/dOmega)' % name)

    print()
    for vname, _ in variants:
        print('%-12s vs make_f: worst abs %.3e   worst rel %.3e   %s'
              % (vname, worst_abs[vname], worst_rel[vname],
                 'ok' if worst_rel[vname] <= TOL_REL else 'FAIL'))
    print()

    def timeit(fun, n=5):
        fun(1.0, sig0, psi0, omg0, dH, delta, True)          # compile
        t0 = time.perf_counter()
        for _ in range(n):
            fun(1.0, sig0, psi0, omg0, dH, delta, True)
        return (time.perf_counter() - t0) / n

    times = [('make_f (reference)', timeit(f_ref))]
    for vname, fun in variants:
        times.append((vname, timeit(fun)))
    base = times[0][1]
    for label, tt in times:
        print('  %-18s %8.2f ms   %5.2fx' % (label, 1e3 * tt, base / tt))
    print()
    return ok


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    all_ok = True
    for d, pkl, arr in DEVICES:
        if not os.path.exists(os.path.join(ROOT, d, pkl)):
            print('skip %s (no %s)' % (d, pkl))
            continue
        all_ok = check_device(d, pkl, arr, reps) and all_ok
    print('=' * 78)
    print('ALL CHECKS PASSED' if all_ok else 'THERE WERE FAILURES')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
