# ThesisZGNR

A 3-ZGNR device from SIESTA (SZ basis), TranSIESTA + TBtrans, pz-only device
orbitals, with a **density-matrix-dependent `dH`** (mean-field `Rep = 2` on the
two mid-ribbon sites `K1, K2 = 23, 24`) and a box-pulse bias on the leads
(smoothed, on at t ~ 10). Adapted from `examples/ThesisExample` (scripts
5a/5b/5d), with 25 Lorentzians per lead instead of the thesis' 81 to keep the
auxiliary-mode count at test size.

Like every test in the suite it runs the same device, drive and start state
through `scipy_ode` (shortlib right-hand side, piecewise) and the two MPI codes
`zand` (orthogonal basis) and `nozand` (nonorthogonal basis, `Bias_NO.py`
transforms both the argument and the result of `dH` — this is a real SIESTA
device, so `S != I`), and prints the difference metrics plus a figure.

## Run

```bash
sh run.sh                  # device build + fit + scipy_ode + zand + nozand + compare
```

or through the suite driver: `./run_zandpack_tests ThesisZGNR`.

## WARNING: the fitted self-energies are not positive semidefinite

The Lorentzian fit produces Γ matrices with **negative eigenvalues** — the fit
prints the min eigenvalue and the "take extra care" warning (this 25-pole fit
has min eig ~ −13; `examples/ThesisExample` shows the same behaviour with its
81-pole fit, so it is not caused by the reduced pole count).

**What this means for the test's purpose:** nothing. The test checks that
`shortlib`/`scipy_ode`, `zand` and `nozand` agree *on the same fitted problem*,
and they consume byte-identical input arrays (`TDZGNR/Arrays/`), so the
cross-code agreement is a valid check regardless of fit quality.

**What it means for the physics:** a non-PSD Γ is not a trustworthy
self-energy. Currents, occupations and transients may be distorted by it, and
a growing mode from a negative eigenvalue can force any propagator into very
small steps — this is (part of) why the Python side needs loose tolerances
(`atol/rtol = 1e-6/1e-4`) and the piecewise method (`ZP_CHUNK = 10`, set in
`run_scipy_ode.py`) to stay inside the memory budget. Do not quote this test's
currents as physical predictions.

## Settings that matter

| setting | value | why |
|---|---|---|
| `NL` (Lorentzians/lead) | 25 | test size (thesis: 81) |
| `tol` (`tofile` / `zero_tol`) | 1e-4 | same cutoff both sides |
| `eps` (MPI, `Initial.py`) | 1e-8 | keep the MPI side converged (SR lesson) |
| `atol/rtol` (scipy) | 1e-6/1e-4 | 1e-8/1e-8 makes RK45 take ~1e-6 steps; the package path then drifts past 30 GB on this device |
| `ZP_CHUNK` | 10 | piecewise propagation, caps the peak at ~5 GB |
| window | t = 0…20 | pulse on at t ~ 10 |

The steady-state preparation of the original example (`TDeq_1.sh`: ts-contour,
`modify_occupations`, `SCF`, `psinought`) is deliberately not part of the
default run — as in the other tests it belongs in a separate `run_steady.sh`
(not yet written for this test).
