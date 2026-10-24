# shortlib — a fast, verifiable re-implementation of `make_f`

`rhs.py` rebuilds the right-hand side of the Zandpack propagation
(`TD_Transport.make_f()`) independently, from the equations in
`Zandpack.equations`, as a numba function with the same signature. It exists so
the tests can propagate with something fast, inspectable and usable with inputs
other than the package's own pickled `TD_Transport`.

**It is now what the tests use.** Every test's Python driver calls
`shortlib.driver.propagate(...)`, which selects the right-hand side from the
environment:

| variable | default | meaning |
|---|---|---|
| `ZP_RHS` | `shortlib` | `shortlib` = `rhs.py`; `make_f` = the package's `TD_Transport.make_f` (the switch back) |
| `ZP_CHUNK` | `1` | eval points integrated per `solve_ivp` call; `> 1` uses the piecewise method (below) |
| `ZP_TOL_TIGHTEN` | `1e-2` | factor the solver tolerances are scaled by when chunking, so the chunking does not cost accuracy |

The switch back is one word: `ZP_RHS=make_f ./run_zandpack_tests --fast`.

```
python smoke_f.py [device]     # seconds: equivalent to make_f? how much faster?
python verify_rhs.py [reps]    # minutes: 3 devices, perturbed states, invariants
```

Both exit non-zero on mismatch, so they can be used as a test.

## Status

verified equivalent to `make_f` on three devices — `synth_dm` (2 orbitals,
1 k-point), `SR` (12 orbitals, Ntot = 87), `scipy_ode_vs_zand` (12 orbitals,
10 k-points) — on the real propagation state, on randomised states (which
exercise the zero entries of `psi`/`omega`) and at two different `dH`s.
Worst relative deviation ~1e-15 (both sides use `fastmath`). `hbar != 1` also
matches, which pins down the `1/hbar` placement — including the package's
double `1/hbar` on the `Pi` term, reproduced deliberately.

Across the whole suite the difference is below what the tests print: run with
`ZP_RHS=shortlib`, the fast suite's SUMMARY table comes out **identical** to the
`make_f` run (446.8 s → 403.0 s, all 10 tests ok).

## Speed (per RHS call, this machine)

| device | n_mode | map fill | `make_f` | `dense_omega=False` (default) | `True` |
|---|---|---|---|---|---|
| synth_dm | 24 | 30.6 % | 0.05 ms | 0.03 ms (1.55×) | 0.02 ms (3.1×) |
| SR | 2088 | 36.0 % | 130.7 ms | **53.9 ms (2.43×)** | 124.1 ms (1.05×) |
| scipy_ode_vs_zand | 840 | 7.5 % | 19.7 ms | **7.9 ms (2.50×)** | 40.0 ms (0.49×) |

`dense_omega=False` — **the default** — keeps the auxiliary-mode matrix packed
and contracts it row by row; `True` densifies it per k-point and gets everything
from matrix products. Both use numpy products throughout; the dense one is
faster only when `n_mode` is tiny, is 2× slower on the 7.5 %-full map, and
allocates four `n_mode²` arrays per call, so it is not the default. (The
absolute `make_f` times swing with machine load — read the ratios.)

## Where the remaining time goes (`profile_rhs.py`)

`profile_rhs.py` switches blocks off by doctoring the index maps (same
arithmetic, `om_ptr` zeroed then `psi_mask` cleared) and times the pieces:

| device | omega block | psi block | `[H,sigma]+Pi` | ns per omega entry |
|---|---|---|---|---|
| SR | 98 % | 1 % | 1 % | 32 |
| scipy_ode_vs_zand | 93 % | 5 % | 2 % | 52 |
| synth_dm | 60 % | 12 % | 28 % | 133 |

The auxiliary-mode block is essentially *all* of the cost on the devices that
matter, so it is the only place worth optimizing further. It is now
**memory-bandwidth bound**, not flop bound: for SR it gathers
`Ixi_k[src]` and `psi_conj[src]` per row, which is `2 · n_om · no · 16 B` ≈
**640 MB read per RHS call**, and the omega block's ~53 ms for that is ≈12 GB/s — within a factor
of two of single-core streaming bandwidth. Measured cost per entry tracks the
row length (32 ns at ~750 entries/row, 52 ns at ~21/row where numpy's per-row
call overhead dominates, 133 ns at ~7/row).

**What would still help, and what it costs.** The gather-based form reads 640 MB
because it re-reads `psi`/`Ixi` rows per entry. A row-blocked gemm form —
`S1b = psi_k[r0:r1] @ Ixi_k.T`, `S2b = xi_k[r0:r1] @ psi_conj.T`, plus a blocked
`Omega_b @ xi_k` for the omega-sum — reads those matrices once (~1 MB) and does
the same arithmetic as small gemms, at the price of `(RB, n_mode)` temporaries
(~2 MB per block at RB = 64) and two scalar scatter passes over the 1.57e6
stored entries. Expect roughly a 2× on the omega block for SR-like fill (36 %),
and *no* gain — possibly a loss — at 7.5 % fill, where the gemm computes 13×
more entries than exist. That is why it is not implemented: the suite-level
payoff is small (device builds dominate, see the main handoff), and the current
form is already 2× the reference.

Two cheap wins that *were* taken after this review: `conj(psi)` hoisted out of
the row loop (one `(n_mode, no)` array per k instead of one `(n_r, no)` per row)
and `chi_m` precomputed as a vector instead of rebuilt per row. That moved SR
70.7 → 53.9 ms (1.95× → 2.43×) and `scipy_ode_vs_zand` 8.9 → 7.9 ms (2.16× →
2.50×), with the results unchanged to 1e-15.

## Piecewise evaluation — `piecewise.py`, `compare_scipy_ode.py`

The package's `scipy_ode` makes **one `solve_ivp` call per eval point** and its
peak memory grows with that count — that is what OOM-kills `SR`. `solve_pieces`
integrates `chunk` eval points per call instead, carrying `(sigma, psi, omega)`
across pieces and keeping only the current; `chunk = 1` is the package's call
pattern exactly, so it is the control.

`compare_scipy_ode.py` runs the four variants, each in its own process so the
peak RSS means something, and diffs them against the baseline:

```
python compare_scipy_ode.py run <device> <variant> <out.npz> [--chunk K] [--t1 T] [--n N]
python compare_scipy_ode.py compare <ref.npz> <other.npz> ...
```

**synth_1, `t = 0…20`, N = 60**

| variant | max\|dJ_l\| | max\|dN\| | secs | peak MB |
|---|---|---|---|---|
| `ref` = scipy_ode + make_f | — | — | 18.8 | 608 |
| scipy_ode + shortlib f | 2.9e-15 | 2.4e-15 | 9.1 | 516 |
| pieces chunk=1 + make_f | **0.0** | **0.0** | 19.0 | 608 |
| pieces chunk=10 + make_f | 5.0e-08 | 3.3e-08 | 18.6 | 608 |

**SR, `t = 0…2`, N = 10**

| variant | max\|dJ_l\| | max\|dN\| | secs | peak MB |
|---|---|---|---|---|
| `ref` = scipy_ode + make_f | — | — | 85.0 | 3658 |
| scipy_ode + shortlib f | 2.0e-15 | 8.9e-16 | 49.5 | 3293 |
| pieces chunk=10 + make_f | 5.7e-06 | 7.7e-07 | 90.3 | 1657 |
| pieces chunk=10 + shortlib f | 5.7e-06 | 7.7e-07 | **38.2** | **1564** |

Two conclusions. **The new `f` reproduces the tests' results** — 2e-15 on both
devices, i.e. the same currents a test would report. And **chunking does bound
the memory**: SR's peak halves, 3658 → 1657 MB, while the current moves by
5.7e-06 (far below the ~1e-4 metrics the tests print). That `chunk=1` reproduces
`scipy_ode` *bit for bit* shows the segment loop is right, so the 5.7e-06 is the
solver restart and not a transcription error.

Best combination measured: `pieces chunk=10 + shortlib f` — **2.2× faster and
2.3× less memory** than `scipy_ode + make_f`.

In the tests this is reached with `ZP_CHUNK=10` (any test, no code change);
`ZP_TOL_TIGHTEN` then tightens the solver tolerances to compensate for the
restart, which is what keeps the deviation at the 1e-6..1e-7 level rather than
growing with the piece count.


* no index triple loops: `H@Psi`, `sigma@xi`, `Psi.T@Ixi`, `Om@xi` and the
  row-wise products are all `@`;
* no allocation inside a mode loop (no `omega_idx[...].copy()`, no `Hchi.copy()`,
  no `sigma[k].copy()`, no full `psi.conj()` copy, no boolean mask plus
  fancy-index temporaries);
* the diagonal shift `[H - I chi] Psi` is `Psi @ Ht.T - chi*Psi`;
* the two static index maps are read once into flat arrays.
