# 14_psd_numba_fusion — targets 1+2 of the numba shortlist

Implements the two top items from the 12_gram_profile shortlist
(profile: psd_solve_admm 51.9% with project_psd 36.5% inside, both
dominated by per-iteration temporaries). Backup:
backups/AI_fit_2026-10-09_pre_psd_numba.py.

## What changed (AI_fit.py)

Four `_njit(cache=True)` kernels with pure-numpy fallbacks
(`NUMBA_DISABLE_JIT=1` respected; `_njit = None` -> the old numpy
arithmetic, reproduced exactly):

| kernel | replaces | allocations killed |
|---|---|---|
| `_psd_hermitize_kernel` | `0.5*(C + conj(swapaxes(C,1,2)))` | 2 temps -> writes straight into `Her` |
| `_psd_reconstruct_kernel` | `einsum("mik,mk,mjk->mij", V, lam_pos, conj(V))` | lam broadcast + full conj(V) copy + einsum internals -> ONE output |
| `_admm_ghat_kernel` | `(Z-U)` full temp, fancy-index copy, `np.where`, `AtY + rho*G` | 4 temps -> one pass into a reused RHS buffer |
| `_admm_c_hat_kernel` + in-place `np.add/np.subtract` | `alpha*C + (1-alpha)*Z`, `U + C_hat - Z_new`, `C_hat + U` | 3 temps -> reused `C_hat`/`S` workspaces |

`psd_solve_admm` now allocates the iteration workspaces (G, C_hat, S)
ONCE; per-iteration only `cho_solve`'s output and `unpack_coefficients`
remain. `project_psd`: ~5 array allocations per call -> 2 (the batched
LAPACK eigh is untouched and is now the entire remaining cost).

## Gates (unit_gate.py; run with numba AND NUMBA_DISABLE_JIT=1)

- ALL FOUR KERNELS **BITWISE-IDENTICAL** to their numpy references,
  at every size tested (incl. M=100, p=27) - even the k-reduction in
  `_psd_reconstruct_kernel` matches numpy's einsum bit-for-bit (its
  loop order coincides); the planned tolerance gate was not needed.
- project_psd new vs pre-change module: BITWISE on random Hermitian,
  random dense, and REAL production residues (clipped counts and
  worst eigenvalues identical).
- psd_solve_admm new vs old on REAL production inputs (both leads,
  50 and 200 iterations): **C rel 0.00e+00** - bit-identical
  trajectories (warm=None start), 1.2-1.3x faster per solve under
  numba.
- Fallback run (NUMBA_DISABLE_JIT=1): ALL PASS, bitwise (the fallback
  reproduces the old arithmetic exactly).
- Ported test suites: ALL PASS / ALL PASS.

## End-to-end (bitwise gate - the strongest possible)

The whole pipeline is bit-identical to the post-refit_scales code,
so the production runs must reproduce their records EXACTLY - and do:

- RSSE gram c0 spot: rms 0.00854132 / 45 poles - identical to the
  post-refit_scales record.
- Ag3Au3 gram production test (6 fits): every start identical
  (rms/n/minEig match the previous run digit for digit), final
  T rms 0.007719 / osc 0.003962 / pair min eig -0.15 - identical;
  total fit wall 244 s -> **204 s** (1.2x).
- Re-profile (same c0 start): profiled wall 42.8 -> **35.1 s**;
  psd_solve_admm 22.2 -> 16.3 s, project_psd 15.6 -> 11.2 s
  (the remainder is the batched LAPACK eigh itself), einsum bucket
  9.0 -> 0.2 s.

## Profile consequence (shortlist correction)

The einsum bucket was almost ENTIRELY project_psd's reconstruction
einsum - NOT gram_and_grad's g-assembly (its 4 einsums measure
~0.2 s total). Shortlist item 3 (fuse the g-assembly) is therefore
DEAD - not worth doing. Post-fusion leaders: the trust-region eigh
(23.5%), project_psd's batched eigh (inside the ADMM, ~28%), the QR
stack (12.4%). These are LAPACK-bound; the remaining levers are
algorithmic (fewer ADMM iterations for ranking, skipping the scale
refit during ranking), not kernel-level.

Cumulative effect of the three performance rounds on the production
gram fit (Ag3Au3, same machine, profiled walls): 110 s (pre) ->
43 s (refit_scales) -> 35.1 s (this round); production 6-fit wall
583 -> 244 -> 204 s; every quality metric unchanged or better
(final T rms 0.00772 vs the stored exact-era 0.0100).

## How to rerun

```
OMP_NUM_THREADS=1 python3 unit_gate.py                     # numba path
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python3 unit_gate.py # fallback
OMP_NUM_THREADS=1 python3 ../12_gram_profile/gram_profile_ag3au3.py
OMP_NUM_THREADS=1 python3 ../12_gram_profile/analyze_prof.py
```
