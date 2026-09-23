#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for FittedSelfEnergy_v2 (basis-table kernel).

Contains all tests from test_FittedSelfEnergy.py (the rewritten evaluate is
checked against the original two-pass opt kernels copied verbatim from
dependencies.py), plus tests for single-point sampling with E randomly
drawn from [-10, 10]:
  E = np.array([(np.random.random() - 0.5) * 20])
both real-valued and with a random imaginary part, with random biases, and
random-length random grids.

Run:  python3 test_FittedSelfEnergy_v2.py
"""
import os
import sys
import time
import numpy as np
import numba as nb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from FittedSelfEnergy import Lorentzian_SE, from_saved_file

FASTMATH = True


# ---------------------------------------------------------------------------
# Reference: original implementation, copied verbatim from dependencies.py
# ---------------------------------------------------------------------------
@nb.njit(cache=False, fastmath=FASTMATH)
def L_sum_opt(E, var, res):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0, :])
    res[:] = 0.0
    ne = len(E)
    W2 = Wi**2
    GW2 = Gi*W2
    for ie in range(ne):
        acc = 0.0 + 0.0j
        e = E[ie]
        for l in range(nl):
            w2l = W2[l]
            A = e - ei[l]
            A*= A
            A += w2l
            A = (GW2[l])/A
            acc +=  A
        res[ie] = acc


@nb.njit(cache=False, fastmath=FASTMATH)
def KK_L_sum_opt(E, var, res):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0, :])
    res[:] = 0.0
    ne = len(E)
    W2 = Wi**2
    GW2 = Gi*W2
    for ie in range(ne):
        acc = 0.0 + 0.0j
        e = E[ie]
        for l in range(nl):
            w2l = W2[l]
            A = e - ei[l]
            KKf = A / Wi[l]
            A*= A
            A += w2l
            A = (GW2[l])/A
            acc +=  (A*KKf)
        res[ie] = acc


@nb.njit(cache=False, fastmath=FASTMATH)
def evaluate_Lorentz_basis_matrix_opt(M, E, ei, gamma, res, tol=1e-15, fact=1.0):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    pars = np.zeros((3, ei.shape[1]), dtype=np.complex128)
    tmp_res = np.zeros(len(E), dtype=np.complex128)

    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                if (np.abs(M[ik, :, i, j]) > tol).any():
                    pars[0] = gamma[ik]
                    pars[1] = ei[ik]
                    pars[2] = M[ik, :, i, j] * fact
                    L_sum_opt(E, pars, tmp_res)
                    res[ik, :, i, j] = tmp_res
    return res


@nb.njit(cache=False, fastmath=FASTMATH)
def evaluate_KK_matrix_opt(M, E, ei, gamma, res, tol=1e-15, fact=1.0):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    pars = np.zeros((3, ei.shape[1]), dtype=np.complex128)
    tmp_res = np.zeros(len(E), dtype=np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                if (np.abs(M[ik, :, i, j]) > tol).any():
                    pars[0] = gamma[ik]
                    pars[1] = ei[ik]
                    pars[2] = M[ik, :, i, j] * fact
                    KK_L_sum_opt(E, pars, tmp_res)
                    res[ik, :, i, j] += tmp_res
    return res


def evaluate_reference(SE, E, bias=0.0, tol=1e-14):
    """The original noopt=False path of Lorentzian_SE.evaluate."""
    Ci, Ei, Gi = SE.Coeffs, SE.Ei, SE.Gi
    res = np.zeros((Ci.shape[0], len(E), Ci.shape[2], Ci.shape[3]), dtype=np.complex128)
    evaluate_Lorentz_basis_matrix_opt(Ci, E - bias, Ei, Gi, res, tol=tol, fact=-0.5j)
    evaluate_KK_matrix_opt(Ci, E - bias, Ei, Gi, res, tol=tol, fact=0.5)
    return res


def gamma_reference(SE, E, bias=0.0, tol=1e-15):
    Ci, Ei, Gi = SE.Coeffs, SE.Ei, SE.Gi
    res = np.zeros((Ci.shape[0], len(E), Ci.shape[2], Ci.shape[3]), dtype=np.complex128)
    evaluate_Lorentz_basis_matrix_opt(Ci, E - bias, Ei, Gi, res, tol=tol, fact=1.0)
    return res


def find_data_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in ('Testing/ABC', 'Testing/ABC/Arrays'):
        p = os.path.join(root, cand)
        if os.path.exists(os.path.join(p, 'num_lorentzians.npy')):
            return p
    raise FileNotFoundError('Testing data (num_lorentzians.npy) not found under ' + root)


# ---------------------------------------------------------------------------
# Tests carried over from test_FittedSelfEnergy.py
# ---------------------------------------------------------------------------
def test_matches_reference():
    SEs = from_saved_file(find_data_dir())
    E_grids = [
        np.linspace(-2.0, 2.0, 500) + 0.001j,
        np.linspace(-5.0, 5.0, 37) - 0.01j,
        np.array([-4.83448646 + 0.0j]),            # single energy point
        np.linspace(-2.0, 2.0, 100),               # real-only E
    ]
    worst = 0.0
    for SE in SEs:
        for E in E_grids:
            for bias in (0.0, 0.3):
                new = SE.evaluate(E, bias=bias, noopt=False)
                ref = evaluate_reference(SE, E, bias=bias)
                assert new.shape == ref.shape
                assert np.allclose(new, ref, atol=1e-10), \
                    'mismatch for bias=%s, ne=%d' % (bias, len(E))
                worst = max(worst, np.abs(new - ref).max())
    print('  max |new - reference| over all cases: %.2e' % worst)


def test_noopt_true_matches_noopt_false():
    SE = from_saved_file(find_data_dir())[0]
    E = np.linspace(-2.0, 2.0, 50) + 0.001j
    a = SE.evaluate(E, noopt=True)
    b = SE.evaluate(E, noopt=False)
    assert np.allclose(a, b, atol=1e-10), np.abs(a - b).max()
    print('  max |noopt=True - noopt=False|: %.2e' % np.abs(a - b).max())


def test_tol_repack():
    SE = from_saved_file(find_data_dir())[0]
    E = np.linspace(-2.0, 2.0, 50) + 0.001j
    for tol in (1e-8, 1e-3, 1e-14):
        new = SE.evaluate(E, tol=tol, noopt=False)
        ref = evaluate_reference(SE, E, tol=tol)
        assert np.allclose(new, ref, atol=1e-10), 'mismatch for tol=%g' % tol
    new = SE.evaluate(E, noopt=False)
    ref = evaluate_reference(SE, E)
    assert np.allclose(new, ref, atol=1e-10)
    print('  repack on tol in (1e-8, 1e-3, 1e-14) consistent with reference')


def test_evaluate_gamma():
    SE = from_saved_file(find_data_dir())[0]
    E = np.linspace(-2.0, 2.0, 50) + 0.001j
    a = SE.evaluate_gamma(E, bias=0.2)
    ref = gamma_reference(SE, E, bias=0.2)
    assert np.allclose(a, ref, atol=1e-10), np.abs(a - ref).max()
    print('  evaluate_gamma matches reference (max diff %.2e)' % np.abs(a - ref).max())


def test_zero_coefficients():
    SE = from_saved_file(find_data_dir())[0]
    Z = Lorentzian_SE(SE.Ei, SE.Gi, np.zeros_like(SE.Coeffs))
    E = np.linspace(-2.0, 2.0, 10) + 0.001j
    out = Z.evaluate(E, noopt=False)
    assert out.shape == (Z.Coeffs.shape[0], len(E), Z.Coeffs.shape[2], Z.Coeffs.shape[3])
    assert np.all(out == 0)
    out2 = Z.evaluate(E, noopt=True)
    assert np.all(out2 == 0)
    print('  all-zero Coeffs: nact=0 handled, output all zeros')


def test_nk_greater_than_one():
    SE0, SE1 = from_saved_file(find_data_dir())
    SE2 = Lorentzian_SE(np.concatenate([SE0.Ei, SE1.Ei]),
                        np.concatenate([SE0.Gi, SE1.Gi]),
                        np.concatenate([SE0.Coeffs, SE1.Coeffs]))
    E = np.linspace(-2.0, 2.0, 50) + 0.001j
    new = SE2.evaluate(E, noopt=False)
    ref = evaluate_reference(SE2, E)
    assert new.shape[0] == 2
    assert np.allclose(new, ref, atol=1e-10), np.abs(new - ref).max()
    assert np.allclose(new[0], SE0.evaluate(E, noopt=False), atol=1e-12)
    assert np.allclose(new[1], SE1.evaluate(E, noopt=False), atol=1e-12)
    print('  nk=2 matches reference and per-lead evaluations')


def test_hermitian_paths_raise():
    SE = from_saved_file(find_data_dir())[0]
    E = np.linspace(-2.0, 2.0, 5) + 0.001j
    for call in (lambda: SE.evaluate(E, hermitian_parts=True),
                 lambda: SE.evaluate_gamma(E, force_hermitian=True)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError('expected NotImplementedError')
    print('  hermitian paths raise NotImplementedError as documented')


# ---------------------------------------------------------------------------
# New tests: random single-point sampling, E in [-10, 10]
# ---------------------------------------------------------------------------
def test_random_single_point_real():
    """E = np.array([(np.random.random() - 0.5) * 20]), real-valued."""
    np.random.seed(42)
    SEs = from_saved_file(find_data_dir())
    worst = 0.0
    for _ in range(25):
        E = np.array([(np.random.random() - 0.5) * 20])
        for SE in SEs:
            new = SE.evaluate(E, noopt=False)
            ref = evaluate_reference(SE, E)
            assert new.shape == ref.shape
            assert np.allclose(new, ref, atol=1e-10), \
                'mismatch at E=%s' % E
            worst = max(worst, np.abs(new - ref).max())
    print('  25 random real single points in [-10, 10]: max diff %.2e' % worst)


def test_random_single_point_complex_and_bias():
    """Single random points with random imaginary part and random bias."""
    np.random.seed(43)
    SEs = from_saved_file(find_data_dir())
    worst = 0.0
    for _ in range(25):
        E = np.array([(np.random.random() - 0.5) * 20]) \
            + 1j * np.array([(np.random.random() - 0.5) * 2.0])
        bias = (np.random.random() - 0.5) * 4.0
        for SE in SEs:
            new = SE.evaluate(E, bias=bias, noopt=False)
            ref = evaluate_reference(SE, E, bias=bias)
            assert np.allclose(new, ref, atol=1e-10), \
                'mismatch at E=%s, bias=%s' % (E, bias)
            worst = max(worst, np.abs(new - ref).max())
    print('  25 random complex single points with bias: max diff %.2e' % worst)


def test_random_grids():
    """Random-length random grids, energies in [-10, 10]."""
    np.random.seed(44)
    SEs = from_saved_file(find_data_dir())
    worst = 0.0
    for _ in range(10):
        ne = np.random.randint(1, 200)
        E = (np.random.random(ne) - 0.5) * 20 + 0.001j
        for SE in SEs:
            new = SE.evaluate(E, noopt=False)
            ref = evaluate_reference(SE, E)
            assert np.allclose(new, ref, atol=1e-10), \
                'mismatch for ne=%d' % ne
            worst = max(worst, np.abs(new - ref).max())
    print('  10 random grids (ne in [1, 200)): max diff %.2e' % worst)


def test_performance():
    SE = from_saved_file(find_data_dir())[0]
    E500 = np.linspace(-2.0, 2.0, 500) + 0.001j
    E1 = np.array([-1.2345 + 0.001j])
    evaluate_reference(SE, E500)
    SE.evaluate(E500, noopt=False)
    SE.evaluate(E1, noopt=False)

    def med(f, n=15):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return np.median(ts)

    t_ref500 = med(lambda: evaluate_reference(SE, E500))
    t_new500 = med(lambda: SE.evaluate(E500, noopt=False))
    t_ref1 = med(lambda: evaluate_reference(SE, E1), n=200)
    t_new1 = med(lambda: SE.evaluate(E1, noopt=False), n=200)
    print('  ne=500 : original %.4f s, v2 %.4f s (%.1fx)'
          % (t_ref500, t_new500, t_ref500 / t_new500))
    print('  ne=1   : original %.1f us, v2 %.1f us (%.1fx)'
          % (t_ref1 * 1e6, t_new1 * 1e6, t_ref1 / t_new1))
    assert t_new500 < t_ref500
    assert t_new1 < t_ref1


# ---------------------------------------------------------------------------
# Additional tests: complex Hermitian Coeffs (M[i,j] = conj(M[j,i]) per
# Lorentzian block). The data in Testing/ABC is real symmetric (verified:
# max|M - M.T| == max|M - M^H| ~ 1e-15), i.e. trivially Hermitian; these
# tests use blocks with nontrivial imaginary parts.
# ---------------------------------------------------------------------------
def _hermitian_blocks(shape, rng):
    """Random complex matrices with Hermitian (no, no) blocks:
    M[..., i, j] = conj(M[..., j, i])."""
    A = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def test_hermitian_coeffs_match_reference():
    """Genuinely complex Hermitian Coeffs evaluate identically to the
    reference implementation (the kernels make no symmetry assumption,
    but this pins it down)."""
    rng = np.random.default_rng(7)
    SEs = from_saved_file(find_data_dir())
    worst = 0.0
    for SE in SEs:
        CH = _hermitian_blocks(SE.Coeffs.shape, rng)
        assert np.allclose(CH, np.conj(np.swapaxes(CH, -1, -2)))
        SEH = Lorentzian_SE(SE.Ei, SE.Gi, CH)
        E_grids = [
            np.linspace(-10.0, 10.0, 50),          # real E over the full range
            np.array([3.3]),                       # single real point
            np.linspace(-2.0, 2.0, 20) + 0.01j,    # complex E
        ]
        for E in E_grids:
            new = SEH.evaluate(E, noopt=False)
            ref = evaluate_reference(SEH, E)
            assert new.shape == ref.shape
            assert np.allclose(new, ref, atol=1e-10), \
                'mismatch for ne=%d' % len(E)
            worst = max(worst, np.abs(new - ref).max())
    print('  Hermitian Coeffs match reference: max diff %.2e' % worst)


def test_hermitian_coeffs_result_structure():
    """For Hermitian Coeffs and real E, the self-energy decomposes into
    Hermitian parts: (SE + SE^dagger)/2 and (SE - SE^dagger)/(2i) must both
    be Hermitian, since B[ie, l] = a + i b with real a, b and M_l = M_l^dagger.
    Also, the packed active set must be symmetric in (i, j), because
    |M[i, j]| = |conj(M[j, i])| = |M[j, i]|."""
    rng = np.random.default_rng(8)
    SE = from_saved_file(find_data_dir())[0]
    CH = _hermitian_blocks(SE.Coeffs.shape, rng)
    SEH = Lorentzian_SE(SE.Ei, SE.Gi, CH)

    # packed active set symmetric in (i, j)
    for Gpack, I, J in SEH._packed:
        pairs = set(zip(I.tolist(), J.tolist()))
        assert all((j, i) in pairs for (i, j) in pairs), \
            'active set not symmetric for Hermitian Coeffs'

    E = np.linspace(-10.0, 10.0, 20)  # real E
    out = SEH.evaluate(E, noopt=False)[0]
    for ie in range(len(E)):
        S = out[ie]
        H1 = 0.5 * (S + S.conj().T)
        H2 = (S - S.conj().T) / (2j)
        assert np.allclose(H1, H1.conj().T, atol=1e-10), \
            'Hermitian part of SE not Hermitian at ie=%d' % ie
        assert np.allclose(H2, H2.conj().T, atol=1e-10), \
            'anti-Hermitian part of SE not Hermitian at ie=%d' % ie
    print('  Hermitian Coeffs: SE splits into Hermitian parts; active set symmetric')


def test_hermitian_coeffs_noopt_and_gamma():
    """noopt=True path and evaluate_gamma also consistent for Hermitian
    Coeffs."""
    rng = np.random.default_rng(9)
    SE = from_saved_file(find_data_dir())[0]
    CH = _hermitian_blocks(SE.Coeffs.shape, rng)
    SEH = Lorentzian_SE(SE.Ei, SE.Gi, CH)
    E = np.linspace(-10.0, 10.0, 15) + 0.001j
    a = SEH.evaluate(E, noopt=True)
    b = SEH.evaluate(E, noopt=False)
    assert np.allclose(a, b, atol=1e-10), np.abs(a - b).max()
    g = SEH.evaluate_gamma(E, bias=0.1)
    gref = gamma_reference(SEH, E, bias=0.1)
    assert np.allclose(g, gref, atol=1e-10), np.abs(g - gref).max()
    print('  Hermitian Coeffs: noopt=True == noopt=False, evaluate_gamma matches')


# ---------------------------------------------------------------------------
# Analytic closed-form tests (independent of the reference implementation:
# the reference and v2 share ancestry, so these check the math itself)
# ---------------------------------------------------------------------------
def test_analytic_single_lorentzian():
    """One nonzero coefficient: res must equal the closed-form fused
    Lorentzian + KK expression at that entry and be exactly zero elsewhere."""
    rng = np.random.default_rng(11)
    nk, Nl, no = 1, 5, 4
    Ei = rng.uniform(-8, 8, (nk, Nl))
    Gi = rng.uniform(0.05, 1.0, (nk, Nl))
    E = rng.uniform(-10, 10, 40) + 1j * rng.uniform(-1, 1, 40)
    for (l0, i0, j0) in [(0, 0, 0), (2, 1, 3), (4, 3, 2)]:
        C = np.zeros((nk, Nl, no, no), dtype=np.complex128)
        c = 1.7 - 0.3j
        C[0, l0, i0, j0] = c
        SE = Lorentzian_SE(Ei, Gi, C)
        out = SE.evaluate(E, noopt=False)
        w, e0 = Gi[0, l0], Ei[0, l0]
        L = w * w / ((E - e0) ** 2 + w * w)
        expected = c * (-0.5j * L + 0.5 * L * (E - e0) / w)
        assert np.allclose(out[0, :, i0, j0], expected, atol=1e-12), \
            'closed form violated at (l,i,j)=(%d,%d,%d)' % (l0, i0, j0)
        mask = np.ones((no, no), dtype=bool)
        mask[i0, j0] = False
        assert np.all(out[0, :, mask] == 0), 'inactive entries must stay zero'
    print('  single-Lorentzian closed form reproduced; inactive entries zero')


def test_analytic_energy_at_centre():
    """E exactly at the (single) Lorentzian centre: L = 1, KK factor = 0,
    so SE = -0.5j * M exactly, for real and complex-typed E."""
    rng = np.random.default_rng(12)
    nk, Nl, no = 1, 1, 6
    Ei = np.array([[0.7]])
    Gi = np.array([[0.3]])
    C = rng.standard_normal((nk, Nl, no, no)) + 1j * rng.standard_normal((nk, Nl, no, no))
    SE = Lorentzian_SE(Ei, Gi, C)
    for E in (np.array([0.7]), np.array([0.7 + 0.0j])):
        out = SE.evaluate(E, noopt=False)
        assert np.allclose(out[0, 0], -0.5j * C[0, 0], atol=1e-13), \
            'E at centre must give -0.5j * M'
    print('  E at centre: SE = -0.5j * M exactly (real and complex E)')


def test_analytic_asymptotic_decay():
    """|E| -> infinity: SE ~ 0.5 * sum_l M_l W_l / (E - ei_l) (the -0.5j L
    term is O(1/E^2)), and the norm decays like 1/|E|."""
    rng = np.random.default_rng(13)
    nk, Nl, no = 1, 6, 5
    Ei = rng.uniform(-8, 8, (nk, Nl))
    Gi = rng.uniform(0.05, 1.0, (nk, Nl))
    C = rng.standard_normal((nk, Nl, no, no)) + 1j * rng.standard_normal((nk, Nl, no, no))
    SE = Lorentzian_SE(Ei, Gi, C)
    for E in (np.array([1e6 + 1e5j]), np.array([-2e6 + 0.5j])):
        out = SE.evaluate(E, noopt=False)[0, 0]
        leading = 0.5 * np.einsum('lij,l->ij', C[0], Gi[0] / (E[0] - Ei[0]))
        assert np.allclose(out, leading, rtol=1e-6, atol=1e-12), \
            'asymptotic SE must match 0.5 * sum_l M_l W_l / (E - ei_l)'
    m1 = np.abs(SE.evaluate(np.array([1e3 + 1e2j]), noopt=False)).max()
    m2 = np.abs(SE.evaluate(np.array([1e6 + 1e5j]), noopt=False)).max()
    assert 0.5e-3 < m2 / m1 < 2e-3, 'SE must decay like 1/|E|'
    print('  asymptotics: leading term 0.5*sum M W/(E-ei) matched; 1/|E| decay')


# ---------------------------------------------------------------------------
# Third, independent implementation: pure numpy broadcasting + tensordot
# ---------------------------------------------------------------------------
def evaluate_numpy(SE, E, bias=0.0, tol=1e-14):
    """Independent evaluation: build B with numpy broadcasting, contract
    with tensordot. No numba, no packed rows, different loop structure."""
    Ci, Ei, Gi = SE.Coeffs, SE.Ei, SE.Gi
    Eb = np.asarray(E) - bias
    nk, Nl, ni, nj = Ci.shape
    res = np.zeros((nk, len(Eb), ni, nj), dtype=np.complex128)
    for ik in range(nk):
        e = Eb[:, None]
        w = Gi[ik][None, :]
        ei = Ei[ik][None, :]
        L = w * w / ((e - ei) ** 2 + w * w)
        B = -0.5j * L + 0.5 * L * (e - ei) / w
        M = Ci[ik]
        act = (np.abs(M) > tol).any(axis=0)
        M = np.where(act[None, :, :], M, 0.0)
        res[ik] = np.tensordot(B, M, axes=([1], [0]))
    return res


def test_third_implementation_numpy():
    """Cross-check v2 against the pure-numpy implementation on the ABC data
    (grids and random single points), Hermitian blocks, large random shapes,
    and zero coefficients."""
    rng = np.random.default_rng(14)
    cases = []
    for SE in from_saved_file(find_data_dir()):
        cases.append((SE, np.linspace(-10, 10, 33) + 0.01j))
        cases.append((SE, np.array([(rng.random() - 0.5) * 20])))
    SE0 = from_saved_file(find_data_dir())[0]
    A = rng.standard_normal(SE0.Coeffs.shape) + 1j * rng.standard_normal(SE0.Coeffs.shape)
    SEH = Lorentzian_SE(SE0.Ei, SE0.Gi, 0.5 * (A + np.conj(np.swapaxes(A, -1, -2))))
    cases.append((SEH, np.linspace(-10, 10, 21)))
    nk, Nl, no = 2, 30, 50
    Ei = rng.uniform(-10, 10, (nk, Nl))
    Gi = rng.uniform(0.02, 1.0, (nk, Nl))
    C = rng.standard_normal((nk, Nl, no, no)) + 1j * rng.standard_normal((nk, Nl, no, no))
    C *= (rng.random((nk, Nl, no, no)) < 0.3)
    cases.append((Lorentzian_SE(Ei, Gi, C), (rng.random(60) - 0.5) * 20 + 0.01j))
    cases.append((Lorentzian_SE(SE0.Ei, SE0.Gi, np.zeros_like(SE0.Coeffs)),
                  np.linspace(-5, 5, 7)))
    worst = 0.0
    for SE, E in cases:
        a = SE.evaluate(E, noopt=False)
        b = evaluate_numpy(SE, E)
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-10), \
            'numpy cross-check failed at ne=%d' % len(E)
        worst = max(worst, np.abs(a - b).max())
    print('  numpy cross-implementation: max diff %.2e over %d cases'
          % (worst, len(cases)))


# ---------------------------------------------------------------------------
# Shape and domain edge cases
# ---------------------------------------------------------------------------
def test_minimal_shapes():
    """Minimal shapes: Nl=1, no=1, down to the 1x1 single-Lorentzian case."""
    rng = np.random.default_rng(21)
    E = np.linspace(-10, 10, 9) + 0.01j
    cases = []
    cases.append(Lorentzian_SE(rng.uniform(-5, 5, (1, 1)), rng.uniform(0.05, 1, (1, 1)),
                               rng.standard_normal((1, 1, 1, 1))
                               + 1j * rng.standard_normal((1, 1, 1, 1))))
    cases.append(Lorentzian_SE(rng.uniform(-5, 5, (1, 7)), rng.uniform(0.05, 1, (1, 7)),
                               rng.standard_normal((1, 7, 1, 1))
                               + 1j * rng.standard_normal((1, 7, 1, 1))))
    cases.append(Lorentzian_SE(rng.uniform(-5, 5, (1, 1)), rng.uniform(0.05, 1, (1, 1)),
                               rng.standard_normal((1, 1, 9, 9))
                               + 1j * rng.standard_normal((1, 1, 9, 9))))
    for SE in cases:
        new = SE.evaluate(E, noopt=False)
        ref = evaluate_reference(SE, E)
        npy = evaluate_numpy(SE, E)
        assert np.allclose(new, ref, atol=1e-10)
        assert np.allclose(new, npy, atol=1e-10)
    print('  minimal shapes (Nl=1, no=1) consistent with reference and numpy')


def test_tol_extremes():
    """tol=0 activates every row with any nonzero entry (including values
    below the default tol); tol=1e10 deactivates everything."""
    rng = np.random.default_rng(22)
    SE0 = from_saved_file(find_data_dir())[0]
    C = np.zeros_like(SE0.Coeffs)
    C[:, :, :10, :10] = rng.standard_normal((1, C.shape[1], 10, 10)) \
                        + 1j * rng.standard_normal((1, C.shape[1], 10, 10))
    C[0, 0, 20, 20] = 1e-18   # nonzero, but far below the default tol
    SE = Lorentzian_SE(SE0.Ei, SE0.Gi, C)
    E = np.linspace(-5, 5, 7) + 0.01j
    new = SE.evaluate(E, tol=0.0, noopt=False)
    ref = evaluate_reference(SE, E, tol=0.0)
    npy = evaluate_numpy(SE, E, tol=0.0)
    assert np.allclose(new, ref, atol=1e-10), 'tol=0 must match reference'
    assert np.allclose(new, npy, atol=1e-10), 'tol=0 must match numpy'
    new = SE.evaluate(E, tol=1e10, noopt=False)
    assert np.all(new == 0), 'tol=1e10 must deactivate everything'
    new = SE.evaluate(E, noopt=False)
    ref = evaluate_reference(SE, E)
    assert np.allclose(new, ref, atol=1e-10), 'repack back to default'
    print('  tol=0 and tol=1e10 handled; repack back to default consistent')


def test_degenerate_lorentzians():
    """Repeated centres and/or broadenings (fully degenerate ei and Gi)."""
    rng = np.random.default_rng(23)
    nk, Nl, no = 2, 8, 6
    E = np.linspace(-10, 10, 11) + 0.01j
    variants = {
        'equal centres': (np.tile(np.array([[0.5]]), (nk, Nl)), rng.uniform(0.05, 1, (nk, Nl))),
        'equal widths': (rng.uniform(-5, 5, (nk, Nl)), np.tile(np.array([[0.3]]), (nk, Nl))),
        'fully degenerate': (np.tile(np.array([[0.5]]), (nk, Nl)), np.tile(np.array([[0.3]]), (nk, Nl))),
    }
    for name, (Ei, Gi) in variants.items():
        C = rng.standard_normal((nk, Nl, no, no)) + 1j * rng.standard_normal((nk, Nl, no, no))
        SE = Lorentzian_SE(Ei, Gi, C)
        new = SE.evaluate(E, noopt=False)
        ref = evaluate_reference(SE, E)
        npy = evaluate_numpy(SE, E)
        assert np.allclose(new, ref, atol=1e-10), name
        assert np.allclose(new, npy, atol=1e-10), name
    print('  degenerate Lorentzians (repeated ei/Gi) consistent')


def test_noncontiguous_input_coeffs():
    """Fortran-order and transposed-view Coeffs must be normalized by
    __init__ and evaluate correctly; the input array must not be modified."""
    rng = np.random.default_rng(24)
    SE0 = from_saved_file(find_data_dir())[0]
    base = rng.standard_normal(SE0.Coeffs.shape) \
        + 1j * rng.standard_normal(SE0.Coeffs.shape)
    E = np.linspace(-5, 5, 9) + 0.01j
    nk, Nl, ni, nj = SE0.Coeffs.shape
    big = rng.standard_normal((nk, Nl, ni, 2 * nj)) \
        + 1j * rng.standard_normal((nk, Nl, ni, 2 * nj))
    views = [np.asfortranarray(base),
             big[:, :, :, ::2]]   # strided view, shape (nk, Nl, ni, nj)
    for C in views:
        assert not C.flags['C_CONTIGUOUS']
        snapshot = C.copy()
        SE = Lorentzian_SE(SE0.Ei, SE0.Gi, C)
        assert SE.Coeffs.flags['C_CONTIGUOUS'], 'stored Coeffs must be C-order'
        assert np.array_equal(C, snapshot), 'input array must not be modified'
        new = SE.evaluate(E, noopt=False)
        ref = evaluate_reference(SE, E)
        assert np.allclose(new, ref, atol=1e-10)
    print('  Fortran-order and view inputs normalized; inputs unmodified')


def test_large_random_shapes():
    """Shapes well beyond the 72x72x44 test data, against the reference
    and the numpy implementation."""
    rng = np.random.default_rng(25)
    nk, Nl, no = 3, 20, 60
    Ei = rng.uniform(-10, 10, (nk, Nl))
    Gi = rng.uniform(0.02, 1.0, (nk, Nl))
    C = rng.standard_normal((nk, Nl, no, no)) + 1j * rng.standard_normal((nk, Nl, no, no))
    C *= (rng.random((nk, Nl, no, no)) < 0.5)
    SE = Lorentzian_SE(Ei, Gi, C)
    E = (rng.random(15) - 0.5) * 20 + 0.01j
    new = SE.evaluate(E, noopt=False)
    ref = evaluate_reference(SE, E)
    npy = evaluate_numpy(SE, E)
    assert np.allclose(new, ref, atol=1e-10)
    assert np.allclose(new, npy, atol=1e-10)
    print('  large random shapes (nk=3, Nl=20, no=60) consistent')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            failed += 1
            print('  FAIL: %s' % e)
        else:
            print('  PASS')
    print('%d/%d tests passed' % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
