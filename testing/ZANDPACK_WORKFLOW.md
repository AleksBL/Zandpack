# Zandpack workflow

Zandpack does time-dependent transport for a device coupled to (semi-infinite)
electrodes: it propagates the density matrix (plus auxiliary modes) in time
while a time-dependent `dH` and lead biases drive the system. This is the
workflow the tests exercise, in the order the code expects it.

The tests exist to check `zand`/`nozand` — specialised solvers for these
particular equations — against an independent codebase: the general-purpose
`scipy_ode` propagator (with the `shortlib` right-hand side, or the legacy
`make_f`). All of them solve the same ODEs; the comparison is only meaningful
because every method starts from the same total state and runs with converged
solver tolerances. See the top of `handoff.md` for the full statement of
purpose, and §4a below for how the common start state is enforced.

## 1. Build the device

```python
from Zandpack.TimedependentTransport import TD_Transport

C = TD_Transport([geom_em, geom_ep], geom_dev, kT_i = [0.025, 0.025], mu_i = [0.0, 0.0])
C.Make_Contour(line, 15, pole_mode = 'JieHu2011')      # complex energy contour
C.Electrodes(semi_infs = ['-a1', '+a1'], pp_folder = '../pp')   # semi-infinite directions
C.make_device(elec_inds = [[0], [tx+1]], k = [1,1,1], k_tbtrans = [1,1,1])

C.run_electrodes(manual_H = [elec, elec])   # omit manual_H to use SIESTA
C.run_device(manual_H = dev_H)
C.read_data(sub_orbital = [...], D_lead_ortho = True)
```

* `Make_Contour(line, N_F)` fixes the complex energy line on which the electrode
  self-energies are sampled and fitted (`N_F` Fermi poles).
* `manual_H=` injects a hand-built `sisl.Hamiltonian`, which keeps a test
  tight-binding (no DFT). Without it, SIESTA/TBtrans are run through
  `siesta_python.siesta_python.SiP`.
* `read_data()` pulls the electrode Gammas/self-energies, device `H`, `S`,
  Löwdin orthogonalisers and the pivot from the TBtrans output.
* `D_lead_ortho=True` (the default) computes the Yan Ho Kwok et al. (2013)
  lead–device orthogonalization correction self-energies (`Sig0`/`Sig1`), which
  change the fitted problem and the equilibrium start DM. Set it to `False`
  when TBtrans samples real k-points (e.g. `kp_tbtrans=[1,5,1]`); keep `True`
  for `nk = 1` (a spin-polarised `nk = 2`, two copies at Γ, also wants `True`).
  Caveat: with `True` on a k-sampled system the Python propagation
  (`make_f` + start state from `Arrays/`) and the MPI codes currently converge
  to *different* trajectories — see the `D_lead_ortho` bullet in
  `handoff.md` §5.
* `pp_folder` defaults to `'../pp'` — i.e. a `pp/` directory next to the run
  directory is required whenever SIESTA is involved.

### Steady-state initialisation (optional)

The cmdtools ship the steady-state helpers: `SCF` (the device SCF),
`psinought` (the auxiliary modes ψ/ω at t₀), `qp_grid`, `pseudo_do`… Running
`psinought` is what makes the propagation start from a good steady state;
without it the propagation starts from the isolated device DM with zero ψ/ω and
needs to be run long enough to relax.

## 2. Fit the self-energies

```python
C.Fit(fact = 2.0, NumL = 16, fit_mode = 'all',
      force_PSD = True, force_PSD_tol = [-0.0, -0.0],
      min_method = 'SLSQP', ebounds = (-2,2), wbounds = (0.1,1.0),
      options = {'maxiter': 500}, alpha_PO = 0.001)
```

The sampled self-energies are replaced by a sum of Lorentzians (with Fermi
poles), which is what makes the later propagation cheap. `force_PSD` pushes the
fitted Γ towards positive semidefiniteness; a negative eigenvalue of the fitted
Γ is what the "extra care" warning is about.

## 3. Prepare the propagation

```python
C.diagonalise()
C.get_propagation_quantities()          # Ixi, propagators in the Löwdin basis
C.get_dense_matrices(zero_tol = 1e-8)   # dense matrices, small elements dropped
C.Check_input_to_ODE(loose_fermi = True)
f = C.make_f()                          # rhs of the ODE: (dσ, dψ, dω)
```

`make_f()` (and the whole Python propagation) works in the **orthogonal**
(Löwdin) basis. The later MPI codes take the same quantities from disk.

## 4a. Propagate in Python

```python
from Zandpack.decoupled_functions import scipy_ode, AdaptiveRK4, AdaptiveDOP

t, dm, jl = scipy_ode(f, C.sigma, C.Psi_vec, C.omegas, t0, t_eval,
                      dH, bias, C.Ixi, dH_given = True,
                      method = 'RK45', atol = 1e-8, rtol = 1e-8)

t, data = AdaptiveRK4(f, C.sigma, C.Psi_vec, C.omegas, eps, t0, t1,
                      dH, bias, C.Ixi, name = 'run')   # data['current_left'], ...
```

* `dH` and `bias` are supplied by the test (`Bias.py`, `shared.py`), in the
  orthogonal basis.
* `scipy_ode` integrates segment by segment and returns one current column per
  k-point (plus the DM of the first k-point only).
* **Start both sides from the same state.** `tofile()` writes the state the MPI
  codes read (`DM_Ortho`, `xi`, `Ixi`, …). To keep the Python propagation
  comparable, load those same arrays *after* `get_dense_matrices()` (which is
  the call that sets `sigma`), then call `get_propagation_quantities()` again so
  `psi`/`omega` are rebuilt from the same `xi`/`Ixi`:

  ```python
  C.get_dense_matrices(tol)
  from Zandpack.Loader import flexload
  A = 'NAME/Arrays/'
  C.sigma = flexload(A + 'DM_Ortho.npy')
  C.xi    = flexload(A + 'xi.npz')
  C.Ixi   = flexload(A + 'Ixi.npz')
  C.get_propagation_quantities()
  ```

  Without this the two runs solve different initial-value problems, and any
  refinement written into `Arrays/` (e.g. by `SCF`/`psinought`) is seen only by
  the MPI side.
* **Sample the Python side densely — but mind the memory.** `scipy_ode` only
  returns the current at its `t_eval` points while the MPI codes write it on
  their adaptive grid; a coarse `t_eval` makes an interpolated comparison look
  far worse than the methods actually are. Extra points cost time *and* memory:
  `scipy_ode` runs one `solve_ivp` per eval point and the peak scales with that
  count (measured on SR at a fixed `t = 0 … 2`: N = 50 → >12 GB, N = 10 →
  3.5 GB). The window contributes too, but N is the cheap lever.
  `ZP_CHUNK=K` integrates K eval points per solver call instead — the piecewise
  method — which is how the tests keep a fine grid without the memory.
* **The right-hand side is switchable.** The tests do not call `scipy_ode`
  directly; they go through `shortlib.driver.propagate(...)`, which builds the
  RHS either from the package (`TD_Transport.make_f`) or from the
  re-implementation in `shortlib/rhs.py` — `ZP_RHS=shortlib` is the default and
  `ZP_RHS=make_f` switches back. Both agree to ~1e-15 (see
  `shortlib/verify_rhs.py`); the alternative is 1.55–2.5× faster per call.
* `AdaptiveRK4` is an embedded RK-Fehlberg integrator with an adaptive step; it
  writes `_times`, `_JL`, `_JR`, `_current_*` and `<name>_last_*.npy` as it goes.
  Its error estimator is currently degenerate (returns 1), so the step only ever
  shrinks — it runs, but it is slow. `Optimized_RK45.AdaptiveRK4_Opti` is the
  same method with an optimised inner loop, but it only accepts the
  `get_dense_matrices_purenp()` + `make_f_purenp()` data path.

## 4b. Propagate with the MPI codes

```python
C.tofile('NAME', tol = 1e-8)      # writes NAME/Arrays/ + NAME/Info.txt
```

```bash
export ZANDPACK_HBAR=1.0 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mpirun -np 3 zand   Dir=$PWD                     # orthogonal basis
mpirun -np 3 nozand Dir=$PWD bias_file=Bias_NO.py  # nonorthogonal basis
```

* Both read `Initial.py` (run length, step, tolerances, `Arrs`) and `Bias.py`
  (`dH`, `bias`, `dissipator`) from the run directory, and write `<name>_save/`.
* **`zand` takes only `Dir=…`** — it does not understand `bias_file=`,
  `initial_file=` or `out_dir=`. To run `zand` with a different bias you have
  to *swap the file* (`cp Bias_saw.py Bias.py`, restore afterwards).
  `nozand` does accept those keywords.
* `nozand` additionally takes `bias_file=`, `initial_file=`, `out_dir=`, and
  reads the optional `compress_mat`, `dm_triu_only`, `dm_occ_only` flags from
  `Initial.py`.
* `zand` is the older code; `nozand` is the preferred one (more sparse → faster).
* One rank acts as the master and broadcasts the Hamiltonian at each step, so
  `-np` must be at least 1 + (number of leads).
* The integrator comes from the **`RK_method` environment variable**
  (`RK45`, `RK78`, `DOP54`, `cash-karp`), read in
  `Zandpack/mpi_runge_kutta_methods.py`; without it the code falls back to
  `Zandpack.k0nfig.RK_Method`, whose default is `DOP54`. Set it explicitly if
  the run is meant to use a particular scheme.
* The codes write the current and the density matrix every `saveevery`
  *substeps*, so a window shorter than that interval leaves the save directory
  empty — the post-processing then has nothing to read.
* **Clear the save directory before every run.** The codes *append*
  `current_<lead>_<n>.npy` / `DM<n>.npy` chunks rather than overwriting, so a
  leftover directory (from an earlier, longer run) is stitched into the new
  result by `J()`/`N()` and silently corrupts the comparison.

## 5. Post-process

```python
from Zandpack.plot import plt, J, N, DM, PI    # plt is pre-styled

t, j = J(['NAME_save_nozand'])   # j[lead] has one column per k-point
t, N = N(['NAME_save_nozand'], S = Hlp.S, insert_tril = True)
```

* `J` (`mpi_tools.combine_currents`) reads the `current_<lead>_<chunk>.npy`
  chunks; `N` (`occupation_number`) reads the DM chunks.
* `J`/`N` are also available as `Zandpack.mpi_tools.combine_currents` etc.

## 6. Bases, and the transformations between them

* Internally (`make_f`, `zand`, `scipy_ode`, `AdaptiveRK4`) the Hamiltonian and
  the density matrix live in the **orthogonal** (Löwdin) basis.
* `nozand` propagates in the **nonorthogonal** basis, so its `dH` must be
  transformed, and its density matrix must be converted to get the particle
  number of the orthogonal representation.

```python
from Zandpack.Help import TDHelper
Hlp = TDHelper('NAME')            # reads NAME/Arrays/
L, iL, S = Hlp.Lowdin, Hlp.invLowdin, Hlp.S     # S^(-1/2), S^(1/2), S

def HamO2NO(H): return iL @ H @ iL      # orthogonal → nonorthogonal
def HamNO2O(H): return L  @ H @ L       # nonorthogonal → orthogonal
def DM_O2NO(dm): return L  @ dm @ L     # same for the density matrix
def DM_NO2O(dm): return iL @ dm @ iL
```

* `Tr[DM]` in the orthogonal basis equals `Tr[S·DM]` in the nonorthogonal
  basis — this is exactly what `N(..., S = Hlp.S)` computes, and it is why
  `nozand`'s occupation numbers are comparable with the orthogonal ones.
* `dH` is a Hamiltonian-like object, so `nozand`'s `Bias.py` uses `HamO2NO`.
* **If `dH` also *reads* `sigma`** — a mean-field / Hartree-like feedback — then
  the *argument* needs a transform too, because `nozand` hands `dH` the density
  matrix in the nonorthogonal basis. Both directions of the workflow input have
  to be converted:

  ```python
  def dH(t, sigma_NO):
      sigma_O = iL @ sigma_NO @ iL          # DM_NO2O
      return iL @ dH_orthogonal(t, sigma_O) @ iL   # HamO2NO
  ```

  With a `sigma`-independent `dH` the inner transform is a no-op, which is why
  most `Bias_NO.py` files only transform the result. On a device with `S = I`
  (`iL = I`) the distinction is invisible either way — a test that wants to
  check it needs a real overlap, as `synth_dm` has.

## 7. Data layout

```
NAME/Arrays/          written by tofile(); consumed by the MPI codes and TDHelper
  H_Ortho.npy         device Hamiltonian (orthogonal basis)
  DM_Ortho.npy        initial density matrix
  S^(-0.5).npy        Löwdin orthogonaliser
  Sig0_*/Sig1_*_O/NO  fitted self-energy (orthogonal / nonorthogonal)
  Ixi.npz, xi.npz     auxiliary-mode couplings
  kT_i, mu_i, num_leads, Positions, pivot, ...

NAME_save/            written by zand / nozand
  current_<lead>_<n>.npy   current chunks (J reads these)
  DM<n>.npy, DMt<n>.npy    density matrix chunks and their times (N reads these)
  PIa*, ...                current/PI matrices when save_PI=True
  NAME.MASTER, NAME.Worker*  per-rank logs
```

`Initial.py` only needs `name`, the time window and tolerances:
`Arrs = load_dictionary(name + '/Arrays/')` is what the codes import.

## 8. Units

`hbar` comes from `Zandpack.td_constants` and can be overridden with the
`ZANDPACK_HBAR` environment variable. All tests set `ZANDPACK_HBAR=1.0`
(natural units); the device scripts print a reminder about this. Every method
in a comparison must see the same `hbar`, otherwise the time scales disagree.
