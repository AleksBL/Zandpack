# Handoff — TestRUN

**Purpose.** To test the `nozand` and `zand` codes against an independent
codebase. In this case the independent codebase is `scipy_ode` — with the
`shortlib` right-hand side (the default now) or the `make_f` implementation
(legacy), both from the Zandpack project.

**Information.** All of the codes solve ODEs. `nozand`/`zand` are specialised
solvers for the particular equations, while `scipy_ode` is a general solver
and gives the independent check on the specialised `zand`/`nozand` codes.

**How this is done.** The scripts in the tests lay out how to set up the
calculations; they produce the input files that are eventually given to the
ODE solvers.

**What we test.** To what degree does `scipy_ode` give the same result in
terms of currents and device occupation? In order for the results from the
different methods to be comparable, the total starting state has to be the
same, and the tolerances of the ODE solvers have to be sufficiently converged.

---

This folder is the Zandpack test suite: small, self-contained tests that run the
*same* device, driving field (`dH`) and bias through different time-propagation
methods and then compare the currents.

Every test compares the currents through `Zandpack.plot.J` (the primary
quantity) and, where available, the trace of the density matrix per k-point
through `Zandpack.plot.N`. Each test writes a figure **and** prints a table of
difference metrics — including a direct `zand - nozand` row, so the two MPI
codes are always compared with each other too. The suite can therefore be read
from the log alone, without opening any plot.

> **Scope — read this first.** Only edit files inside this `TestRUN/` folder.
> The Zandpack package itself (`…/Code/PythonModules/Zandpack/Zandpack/`),
> `siesta_python`, `sisl` and everything else outside `TestRUN/` is read-only:
> inspect it, never modify it. When a test needs different behaviour from the
> package (a bug, an unsupported option, a wrong default), work around it inside
> the test — as the tests here do — or record it, rather than patching the
> package. This is the directive from `AGENTS.md` and it is not negotiable for
> convenience.

---

## 1. Running the suite

```bash
./run_zandpack_tests                      # whole suite, ~34 min, log in ./logs/
./run_zandpack_tests --fast               # smoke-test mode, ~9 min
./run_zandpack_tests ./logs Croy2009      # a subset
./run_zandpack_tests --fast ./logs Croy2009
ZP_JOBS=1 ./run_zandpack_tests            # one test at a time (lowest memory)
```

A plain run is the reference: every test at its own window and grid, which takes
~34 min with `ZP_JOBS=1` (see §3 for the per-test split). The heaviest Python
side observed is `examples/small_bound` at its full window (~11 GB with the
`make_f` right-hand side before the piecewise work; see §5 for the current
numbers).

The driver runs each test's `run.sh` **concurrently** — 3 at a time by default,
`ZP_JOBS` to change it — in its own directory, and collects the metric tables
they print. Tests whose Python side needs a lot of memory run on their own (see
`HEAVY` in the driver). Live progress goes to stdout only; the log file is
written **in the declared test order** (a block is held back until the tests
before it have finished) so two logs can be diffed directly even though the
tests finish out of order:

```
logs/run_fast_2026-10-17_08-38-42.log        # --fast, the reference run (11 tests, ZP_RHS=shortlib)
logs/run_2026-10-17_08-51-53.log             # full run, the reference (11 tests, ZP_RHS=shortlib)

logs/run_fast_2026-10-17_00-18-20.log        # the ten-test pair that compares the two
logs/run_2026-10-17_00-28-16.log             #   right-hand sides (ZP_RHS=shortlib) ...
logs/run_fast_2026-10-16_14-54-35.log        #   ... against (ZP_RHS=make_f)
logs/run_2026-10-16_15-04-16.log             #   -- these four are the equivalence evidence
```

The `make_f` pair is kept on purpose: it is the comparison that justified making
`shortlib` the default. Diffing the two ten-test runs, both SUMMARY tables come
out identical to the printed precision while the wall clock differs (403.0 vs
446.8 s fast, 1520 vs 1745 s full) — the two right-hand sides cannot be told
apart at the resolution the tests report. Those four logs predate `SR` joining
the suite, which is why they are ten-test runs while the two above are the
eleven-test reference.

The log holds the date/time, host, the test list, the mode, the `RK_method`, the
number of concurrent jobs, and then per test the status, the elapsed time and
its metrics, followed by a summary table with one row per method pair and the
total wall clock. The timestamp is what lets a regression be pinned to a run —
diff two logs to see what changed.

A single test is run the same way:

```bash
cd scipy_ode_vs_zand && sh run.sh
```

Environment (every `run.sh` sets the first five; the driver passes them too):

```bash
export ZANDPACK_HBAR=1.0     # natural units, must be the same for all methods
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg        # headless plotting
export RK_method=RK45        # MPI propagator; the package default is DOP54
```

The Python right-hand side is chosen separately (`shortlib/driver.py`, see §5):
`ZP_RHS=shortlib` is the default, `ZP_RHS=make_f` switches back to the package's
`make_f`, `ZP_CHUNK=K` chunks the solver calls, and `ZP_TOL_TIGHTEN` compensates
for that chunking.

`run.sh` does the same four steps:

```
<python: device + fit + scipy_ode>                  →  <name>_save_zand / _nozand
rm -rf <name>_save <name>_save_zand
mpirun -np N zand   Dir=$PWD                       (orthogonal basis)
mv <name>_save <name>_save_zand
rm -rf <name>_save <name>_save_nozand
mpirun -np N nozand Dir=$PWD bias_file=Bias_NO.py  (nonorthogonal basis)
mv <name>_save <name>_save_nozand
python compare.py
```

Both the save directory the code writes into and the one it is moved to are
cleared first. The `rm -rf` before the `mv` stops the new directory being nested
inside the old one; the `rm -rf` before the `mpirun` matters just as much,
because the MPI codes *append* chunks and would otherwise leave stale ones for
`J()`/`N()` to stitch into the result (§5).

### The environment

* `python` is `/home/aleks/oak/bin/python` (3.14); `siesta`, `tbtrans` and
  `mpirun` are on `PATH`; the Zandpack plot style needs LaTeX, which is
  installed. `MPLBACKEND=Agg` is required for headless plotting.
* **18 CPUs and 30 GB of RAM.** `nproc` reports 1, but that is a red herring:
  `OMP_NUM_THREADS=1` is exported in the ambient environment and GNU `nproc`
  honours it, so `nproc` prints the thread count, not the core count.
  `/proc/cpuinfo` and `os.cpu_count()` both report 18. This is what lets the
  driver run several tests at once (see below).
* **`scipy_ode` memory grows with the number of eval points (`ZP_NEVAL`).** It
  runs one `solve_ivp` per eval point and the peak scales with that count — the
  controlled measurement is `SR` at a fixed window (`t = 0 … 2`): **N = 50 →
  >12 GB, N = 10 → 3.5 GB**. The window matters too (it sets how many steps are
  taken in total, and the total also drifts up during a run), but N is the lever
  that is cheap to turn, and the two were confounded in an earlier reading of
  this note. `scipy_vs_ARK4_Zand` and `SR` are the two tests where it bites:
  both sit in `HEAVY` (so they run alone) and both have a small `ZP_NEVAL`
  default. `ZP_CHUNK` now provides the structural fix — see §7 item 7.
* The shell is **zsh**: a glob that matches nothing aborts the *entire* command
  line, so a cleanup like `rm -rf a.* b.*` silently does nothing if `b.*` is
  absent. Verify with `ls`/`find` afterwards.
* Tooling: invoking `mpirun` or `rm -rf` is held for confirmation and needs a
  repeat of the same call; `pkill -f "<pattern>"` can match its own command line
  and kill the caller.
* Long runs belong in the background (`tools.process.start`) — `device.py`,
  `scipy_ode` on the big devices, and each MPI stage all exceed a foreground
  timeout.

### Fast mode

`--fast` exports six environment variables that the tests read; they shorten the
simulation and coarsen the sampling, while the systems still start from the
default (non steady-state) state:

| variable | meaning | fast value |
|---|---|---|
| `ZP_T0`, `ZP_T1` | window (read by `Initial.py` and by the Python driver) | 0, 2 |
| `ZP_NEVAL` | `scipy_ode` sampling points (Python driver) | 50 |
| `ZP_SAVEEVERY` | how often the MPI codes write current/DM (`Initial.py`) | 5 |
| `ZP_ATOL`, `ZP_RTOL` | `scipy_ode` solver tolerances (Python driver) | 1e-6, 1e-4 |

Two tests override those (`FAST_BY_TEST` in the driver), both because their
Python side is memory-hungry: `scipy_vs_ARK4_Zand` runs `t = 0 … 1` with 10
points (a 0–2 window grows past 20 GB), and `SR` runs `t = 0 … 2` with 10 points
and `ZP_CHUNK=10` (one solver call for the window, ~1.6 GB instead of >12 GB at
the global 50 points). 50 points is a deliberate compromise: `scipy_ode` gets
slower *and* needs more memory with every extra point, and the fast mode is a
smoke test — the metric is looser than in a full window, but every pipeline step
and every comparison still runs.

The whole suite finishes in **~9 minutes** (`ZP_JOBS=3`). The current reference
run is `logs/run_fast_2026-10-17_08-38-42.log`, **550.7 s** in total:

| test | elapsed (under 3-way concurrency) | status |
|---|---|---|
| SR | 165.3 s | ok (runs alone, chunked grid) |
| scipy_vs_ARK4_Zand | 99.1 s | ok (runs alone) |
| Croy2009 | 53.0 s | ok |
| PopescuCroy2016 | 78.5 s | ok |
| Honeychurch2018 | 62.9 s | ok |
| scipy_ode_vs_zand | 122.7 s | ok |
| scipy_ode_vs_AdaptiveRK4 | 131.9 s | ok |
| synth_1 | 117.0 s | ok (both drivings) |
| synth_dm | 54.9 s | ok (non-linear dH) |
| examples/small_bound | 91.4 s | ok |
| examples/SuperCond | 67.8 s | ok |

The individual times vary by a few seconds between runs because three tests share
the machine; the total is what matters.

Two things dominate the remaining cost, and fast mode does not shorten either:
the *device build* (SIESTA/TBtrans + the fit), and the ~25 s fixed cost of every
Python propagator process (import + numba JIT + load + building the right-hand
side), which each test pays once or twice. If more headroom is ever needed, the
next levers are the device/fit part (coarser contour, fewer Lorentzians, smaller
k-grids) and dropping `synth_1`'s sawtooth stage from the fast mode.

Tests are kept below ~50 orbitals. `Rib` (694 orbitals) needs more than one
workstation core and is deliberately left out of the driver, as is `old/`
(archive). `synth_1/` is the synthetic bench: a hand-built chain of `nsite` sites
with one orbital each; everything that sets the problem size is in the tuning
block at the top of its `device.py` (`nsite`, `t_hop`, `e_onsite`, `kT`, contour
points, Fermi poles, Lorentzians, tolerance). No DFT and no k-point sampling, so
it runs in ~1 minute (both drivings): the smooth drive and, via
`run_sawtooth.sh`, a sawtooth pulse (`shared.sawtooth`, a ramp whose reset is
smeared over `edge` so the value stays continuous). `run_steady.sh` is the same
test as a separate steady-state pipeline (`device.py` → `SCF` + `psinought` →
both propagators).

## 2. Tests

| Directory | Python propagator | MPI codes | Fast mode |
|---|---|---|---|
| `Croy2009/` | scipy_ode + AdaptiveRK4 | zand + nozand | ok, 53 s |
| `PopescuCroy2016/` | scipy_ode + AdaptiveRK4 | zand + nozand | ok, 79 s |
| `Honeychurch2018/` | scipy_ode + AdaptiveRK4 | zand + nozand | ok, 63 s |
| `scipy_ode_vs_zand/` | scipy_ode | zand + nozand | ok, 123 s |
| `scipy_ode_vs_AdaptiveRK4/` | scipy_ode + AdaptiveRK4 | zand + nozand | ok, 132 s |
| `scipy_vs_ARK4_Zand/` | scipy_ode (ARK4 replaced) | zand + nozand | ok, 99 s (runs alone, short window) |
| `SR/` | scipy_ode | zand + nozand | ok, 165 s (runs alone; chunked grid) |
| `synth_1/` | scipy_ode | zand + nozand | ok, 117 s (both drivings) |
| `synth_dm/` | scipy_ode | zand + nozand | ok, 55 s (**non-linear** dH) |
| `ThesisZGNR/` | scipy_ode (piecewise) | zand + nozand | ok (~7 min; **non-linear** dH, real SIESTA device) |
| `examples/small_bound/` | scipy_ode | zand + nozand | ok, 91 s |
| `examples/SuperCond/` | scipy_ode | zand + nozand | ok, 68 s |
| `Rib/` | scipy_ode | zand + nozand | excluded (694 orbitals) |
| `old/` | — | — | archive |

Five notes about this list:

* `synth_dm` is the **non-linear** test: its `dH` reads the density matrix (a
  mean-field feedback of the `examples/ThesisExample/Bias.py` kind), so the
  right-hand side depends on `sigma` and the four propagators are only
  comparable if they all evaluate it at the same Runge-Kutta stage. They do —
  the Python right-hand side (`shortlib` or `make_f`, both call `dH` the same
  way) and both MPI codes pass the stage density matrix. It is
  also the only synthetic device with `S != I`, which is what makes the
  `Bias_NO.py` argument transform observable (see §5). One reading of its
  numbers is worth knowing up front: `zand − nozand` is 2.3e-05 here, not the
  1e-14 seen on the orthogonal devices, and re-running it with the feedback
  switched off (`Rep = 0` in `shared.py`) gives 2.3e-05 again — so that residual
  belongs to the nonorthogonal path, not to the feedback.
* **`Honeychurch2018` needs attention — flag it.** Its Python `AdaptiveRK4` sits
  ~1.4e-03 (fast) / 1.8e-03 (full) away from the MPI codes, an order of magnitude
  more than any other clean test. Adding a `scipy_ode` run to it settles who the
  outlier is: `scipy_ode` agrees with `zand`/`nozand` to 2.4e-04 (rms 8.8e-05)
  while `AdaptiveRK4` is 1.4e-03 (rms 3.8e-04) away — so the Python ARK4 is the
  odd one out, not the MPI codes. It is also the test whose window (~297 time
  units) is 15× longer than any other, so its Python side needs the denser grid
  it now defaults to (§3.4). Anyone touching the Python propagators should start
  here.
* `AGENTS.md`'s split into "currently working" and "has an overhaul to make it
  work" is now **historical**. All twelve tests in the table above run and print
  metrics, `SR` included; `Rib` needs more than one core and is the only one left
  out, `old/` is an archive. Don't re-do the overhaul work from that file.
* `examples/` also contains many legacy example directories
  (`AuBreakJunction`, `Pohl`, `Pohl2`, `C60`, `GroH`, `GrRibGr`, `Hubbard`,
  `TB_ribbon`, `Testing`, `VerysmallRSSE`, `ZGNR+N`, `AGNR+Tip`,
  `ThesisExample`, `misc`, `generic_wrapper`, `nb_tutorial_*`, plus
  `Reproduce_and_checks/` which mirrors these tests). Only `small_bound` and
  `SuperCond` were taken up; the rest are untested and out of scope unless
  asked for.
* `ZANDPACK_WORKFLOW.md` is the companion document: it describes the Zandpack
  API workflow these tests exercise — which call builds/reads what, which basis
  each propagator works in, the Löwdin transforms, the data layout, and the
  `zand`/`nozand` invocation details.

## 3. Results

`max|dJ|` / `rms|dJ|` are the largest value and the rms of `|reference − other|`
on the overlapping time window of the left current (summed over k-points);
`N_end` is the particle number at the last step. The differences are evaluated
**at the comparator's time points** — for `scipy_ode − zand` that is the MPI
code's adaptive grid, ~25× denser than scipy_ode's `t_eval`, with the
scipy_ode curve linearly interpolated onto it (`np.interp`). The comparison is
never resampled onto a common third grid, so a coarse reference grid inflates
the metric even when the methods agree (§3.3).

### Fast mode (t = 0 … 2, 50 sampling points)

Source: `logs/run_fast_2026-10-17_08-38-42.log` (550.7 s, `ZP_JOBS=3`, `ZP_RHS=shortlib`).

| test | pair | max\|dJ\| | rms\|dJ\| |
|---|---|---|---|
| Croy2009 | scipy_ode − AdaptiveRK4 | 1.30e-03 | 7.78e-04 |
| Croy2009 | scipy_ode − zand / − nozand | 5.07e-05 | 2.40e-05 |
| Croy2009 | zand − nozand | 1.52e-13 | 7.37e-14 |
| PopescuCroy2016 | scipy_ode − AdaptiveRK4 | 1.80e-07 | 1.03e-07 |
| PopescuCroy2016 | scipy_ode − zand / − nozand | 3.98e-09 | 1.63e-09 |
| PopescuCroy2016 | zand − nozand | 1.25e-16 | 4.53e-17 |
| Honeychurch2018 | scipy_ode − AdaptiveRK4 | 1.61e-03 | 4.57e-04 |
| Honeychurch2018 | scipy_ode − zand / − nozand | 2.38e-04 | 8.78e-05 |
| Honeychurch2018 | AdaptiveRK4 − zand / − nozand | 1.43e-03 | 3.80e-04 |
| Honeychurch2018 | zand − nozand | 4.08e-13 | 1.30e-13 |
| scipy_ode_vs_zand | scipy_ode − zand | 2.87e-03 | 1.01e-03 |
| scipy_ode_vs_zand | scipy_ode − nozand | 2.88e-03 | 1.01e-03 |
| scipy_ode_vs_zand | zand − nozand | 1.48e-05 | 6.08e-06 |
| scipy_ode_vs_AdaptiveRK4 | scipy_ode − AdaptiveRK4 | 1.01e-04 | 5.75e-05 |
| scipy_ode_vs_AdaptiveRK4 | scipy_ode − zand / − nozand | 4.65e-06 | 1.90e-06 |
| scipy_ode_vs_AdaptiveRK4 | zand − nozand | 5.61e-15 | 2.74e-15 |
| scipy_vs_ARK4_Zand \* | scipy_ode − zand | 2.96e-01 | 1.02e-01 |
| scipy_vs_ARK4_Zand \* | scipy_ode − nozand | 2.83e-01 | 1.04e-01 |
| scipy_vs_ARK4_Zand \* | zand − nozand | 1.68e-02 | 6.81e-03 |
| synth_1 | scipy_ode − zand / − nozand | 2.18e-05 | 8.53e-06 |
| synth_1 | zand − nozand | 1.06e-14 | 4.05e-15 |
| synth_1 (sawtooth) | scipy_ode − zand / − nozand | 9.05e-05 | 3.27e-05 |
| synth_1 (sawtooth) | zand − nozand | 9.74e-14 | 3.02e-14 |
| synth_dm | scipy_ode − zand | 1.24e-04 | 6.14e-05 |
| synth_dm | scipy_ode − nozand | 1.23e-04 | 6.09e-05 |
| synth_dm | zand − nozand | 2.30e-05 | 9.48e-06 |
| SR \* | scipy_ode − zand / − nozand | 2.68e-01 | 1.26e-01 |
| SR \* | zand − nozand | 2.96e-14 | 1.42e-14 |
| examples/small_bound | scipy_ode − zand / − nozand | 7.26e-04 | 3.26e-04 |
| examples/small_bound | zand − nozand | 2.89e-14 | 1.49e-14 |
| examples/SuperCond | scipy_ode − zand / − nozand | 8.45e-04 | 3.75e-04 |
| examples/SuperCond | zand − nozand | 5.59e-15 | 1.30e-15 |

\* `scipy_vs_ARK4_Zand` runs `t = 0 … 1` with 10 points, so its `scipy_ode` rows
are not comparable with the rest; its `zand − nozand` row is also looser
(three leads, k-sampled, very coarse grid). `SR` runs `t = 0 … 2` with 10 points
and `ZP_CHUNK=10` (one solver call for the whole window), which is what keeps it
inside the memory budget — that is why its `scipy_ode` rows are loose while its
`zand − nozand` row is machine precision.

These are *fast-mode* numbers: a 0–2 window with 50 sampling points. They are
not comparable with the full-window numbers below, and the short smooth window
flatters the sampling (§3.3).

### Full windows (each test's own window and grid)

Source: `logs/run_2026-10-17_08-51-53.log` — the whole suite run **serially**
(`ZP_JOBS=1`, so only one test's memory is live at a time), **2032 s** in total,
every test `ok`. This table is the comparable companion of the fast table above.

| Test (window) | pair | max\|dJ\| | rms\|dJ\| | N_end (ref / other) |
|---|---|---|---|---|
| Croy2009 (t = −10…40) | scipy_ode − AdaptiveRK4 | 6.84e-04 | 1.49e-04 | — |
| Croy2009 | scipy_ode − zand / − nozand | 6.50e-04 | 1.37e-04 | — |
| Croy2009 | zand − nozand | 1.01e-12 | 2.64e-13 | — |
| PopescuCroy2016 (t = 0…200) | scipy_ode − AdaptiveRK4 | 2.10e-02 | 2.58e-03 | — |
| PopescuCroy2016 | scipy_ode − zand / − nozand | 2.10e-02 | 2.59e-03 | — |
| PopescuCroy2016 | zand − nozand | 3.02e-13 | 3.98e-14 | — |
| Honeychurch2018 (t = −147…150) | scipy_ode − AdaptiveRK4 | 1.81e-03 | 9.62e-04 | 1.2949 / — |
| Honeychurch2018 | scipy_ode − zand / − nozand | 1.33e-03 | 8.25e-05 | 1.2949 / 1.293 |
| Honeychurch2018 | AdaptiveRK4 − zand / − nozand | 1.80e-03 | 9.17e-04 | — |
| Honeychurch2018 | zand − nozand | 3.79e-13 | 5.52e-14 | — |
| scipy_ode_vs_zand (t = 0…20) | scipy_ode − zand | 1.72e-02 | 2.32e-03 | 5.9996 / 5.9997 |
| scipy_ode_vs_zand | scipy_ode − nozand | 1.72e-02 | 2.29e-03 | 5.9996 / 5.9996 |
| scipy_ode_vs_zand | zand − nozand | 1.48e-05 | 2.52e-06 | — |
| scipy_ode_vs_AdaptiveRK4 (t = 0…60) | scipy_ode − AdaptiveRK4 | 1.42e-03 | 2.65e-04 | 7.3054 / 7.3049 |
| scipy_ode_vs_AdaptiveRK4 | scipy_ode − zand / − nozand | 1.22e-03 | 2.03e-04 | 7.3054 / 7.1758 |
| scipy_ode_vs_AdaptiveRK4 | zand − nozand | 3.44e-13 | 6.86e-14 | — |
| scipy_vs_ARK4_Zand (t = 0…1) \* | scipy_ode − zand | 2.95e-01 | 1.02e-01 | 16.6664 / 16.6653 |
| scipy_vs_ARK4_Zand \* | scipy_ode − nozand | 2.83e-01 | 1.04e-01 | 16.6664 / 16.6552 |
| scipy_vs_ARK4_Zand \* | zand − nozand | 1.68e-02 | 6.81e-03 | — |
| synth_1 (t = 0…20) | scipy_ode − zand / − nozand | 1.30e-04 | 3.02e-05 | 0.8599 / 0.8603 |
| synth_1 | zand − nozand | 2.25e-14 | 6.68e-15 | — |
| synth_1 (sawtooth) | scipy_ode − zand / − nozand | 7.39e-04 | 2.34e-04 | 0.9041 / 0.9041 |
| synth_1 (sawtooth) | zand − nozand | 7.63e-06 | 1.76e-06 | — |
| synth_dm (t = 0…20) | scipy_ode − zand | 7.49e-04 | 2.11e-04 | 0.9963 / 0.9797 |
| synth_dm | scipy_ode − nozand | 7.58e-04 | 2.02e-04 | 0.9963 / 0.9962 |
| synth_dm | zand − nozand | 2.31e-05 | 6.43e-06 | 0.9797 / 0.9962 |
| SR (t = 0…20, N = 20) | scipy_ode − zand / − nozand | 1.04e+00 | 2.36e-01 | 5.841 / 5.8433 |
| SR | zand − nozand | 2.19e-13 | 8.96e-14 | — |
| ThesisZGNR (t = 0…20, N = 200) | scipy_ode − zand | 8.36e-02 | 1.08e-02 | 22.364 / 22.363 |
| ThesisZGNR | scipy_ode − nozand | 8.35e-02 | 1.08e-02 | 22.364 / 22.351 |
| ThesisZGNR | zand − nozand | 2.50e-04 | 4.37e-05 | — |
| examples/small_bound (t = 0…20) | scipy_ode − zand / − nozand | 4.57e-03 | 6.87e-04 | 10.7176 / 10.6828 |
| examples/small_bound | zand − nozand | 2.89e-14 | 5.98e-15 | — |
| examples/SuperCond (t = 0…20) | scipy_ode − zand / − nozand | 5.06e-03 | 1.04e-03 | 4.0244 / 4.0211 |
| examples/SuperCond | zand − nozand | 9.02e-15 | 2.45e-15 | — |

\* `scipy_vs_ARK4_Zand` runs `t = 0 … 1` (see §1); its Python side cannot take a
longer window, so its two rows are not comparable with the rest.

Per-test times at full window (`ZP_JOBS=1`): SR 506.7 s, scipy_ode_vs_zand 397.5 s,
PopescuCroy2016 226.9 s, scipy_ode_vs_AdaptiveRK4 195.9 s, small_bound 184.7 s,
Honeychurch2018 150.5 s, scipy_vs_ARK4_Zand 127.2 s, synth_1 79.6 s,
SuperCond 65.5 s, Croy2009 49.8 s, synth_dm 47.4 s. **SR's `scipy_ode` row
above is the coarse-grid interpolation effect, not a method disagreement** —
pointwise at well-chosen times it agrees with the MPI codes to 4.0e-04 (see
the SR investigation in §3.3). Note those times predate the `eps = 1e-8`
change; SR now takes ~913 s in the suite.

**Reconciliation with the previous numbers.** This table is at each test's
shipped grid (N = 200 points). The rows previously recorded here came from
*denser* runs (N = 1000–2000 for the §3.3 study), which is why some of them were
10–100× smaller — e.g. `synth_1` 1.10e-06 @ N = 2000 vs 1.30e-04 @ N = 200, and
`scipy_ode_vs_zand` 6.93e-04 @ N = 1000 vs 1.72e-02 @ N = 200. Nothing changed
except the sampling; §3.3 is the bridge between the two. The only row with no
counterpart here is `synth_1`'s steady-state pipeline, which is a separate
script (`run_steady.sh`), not part of the default run.

`zand` and `nozand` agree with each other to machine precision in every test
(`1e-12` … `1e-16`) **except** three:

* `scipy_vs_ARK4_Zand`, 1.7e-02 — its grid is a very coarse 10 points over
  `t = 0 … 1` (and it has three leads).
* `synth_1`'s sawtooth drive, 7.6e-06 — its `nozand` occupation ends 1 % away
  from `zand`/`scipy_ode` (0.8954 vs 0.9041). The smooth drivings are clean, so
  that pair is the one to look at first if the sawtooth numbers ever need to be
  trusted.
* `synth_dm`, 2.3e-05, with `N_end` spread over ~1 % (0.9963 / 0.9797 / 0.9962).
  This is the non-orthogonal device, and the control says it is the *basis*, not
  the feedback: re-running with the mean-field term switched off
  (`Rep = 0` in `shared.py`) gives the same 2.5e-05 and the same ~1 % spread.
  Nothing in the suite explains it further; it is the first thing to look at if
  the nonorthogonal path is ever investigated.

The numbers are **reported, not asserted**; the suite makes no pass/fail claim.

The `ThesisZGNR` rows come from its first standalone run
(`ThesisZGNR/ThesisZGNR_run_2026-10-24_15-45.log`), not from a suite run —
the test joined the driver after the reference logs above were written, so a
`./run_zandpack_tests` run will add it (and the `--fast` table in §1 has no
ThesisZGNR row yet either).

### 3.3 What limits scipy_ode − zand/nozand
`scipy_ode` returns the current only at its `t_eval` points, while the MPI codes
write it on their own adaptive grid (usually 25× denser). Interpolating a
coarsely sampled oscillating current onto that dense grid inflates `max|dJ|`, so
the eval grid is the first thing to look at. Measured, after densifying it:

| test | eval grid before → after | max\|dJ\| before → after |
|---|---|---|
| scipy_ode_vs_zand | 200 → 1000 | 1.72e-02 → 6.93e-04 |
| examples/SuperCond | 100 → 1000 | 1.85e-02 → 2.18e-04 |
| scipy_ode_vs_AdaptiveRK4 | 200 → 1000 | 1.22e-03 → 4.71e-05 |
| synth_1 | 100 → 2000 | 5.06e-04 → 1.10e-06 |
| examples/small_bound | 100 → 200 | 1.86e-02 → 4.37e-03 |

The two largest entries in the full-window table (`scipy_vs_ARK4_Zand` 2.95e-01,
`SR` 1.04) are exactly the tests whose Python side has the coarsest grid
(Δt ≈ 0.1 and 1.05: 10 points over `t = 0…1`, and N = 20 over `t = 0…20`).
For `scipy_vs_ARK4_Zand`, evaluating the same two curves *at the sample points*
instead of interpolating onto the dense grid gives 2.37e-02 — 120× smaller — so
~99% of that number is interpolation artefact. `synth_1/sweep.py` automates both
halves of this study.

**SR's story, told properly (2026-10-24, after one false start).** SR's 1.04 in
the full-window table is dominated by the **coarse Python eval grid** — the
same interpolation effect as the other rows, and the original explanation in
this section was right. Two claims made along the way were wrong and are
recorded in §6. What the measurements actually show:

* Pointwise at well-chosen sample times (`ZP_TLIST=0.1,0.5,1,2,5,10,15,20`,
  one `solve_ivp`, 1.1 GB), in the suite environment (`ZANDPACK_HBAR=1.0`),
  scipy_ode agrees with `zand`/`nozand` to **max|dJ| = 4.0e-04** — in line
  with the clean tests. Occupations agree to ~0.002 (5.841 vs 5.843).
* The MPI controller tolerance contributes little but measurably:
  `eps = 1e-5 → 1e-8` in `Initial.py` moves zand's current by max 0.059 /
  rms 0.007. `eps = 1e-8` is now the shipped value; it roughly doubles SR's
  suite time (467 → 913 s, now the slowest test). Revert to `1e-5` if that
  cost ever matters more than the last digit.
* The false middle chapter: a first investigation "found" that eps was the
  whole story (a 1.67 move) — but those manual runs **lacked
  `ZANDPACK_HBAR=1.0`**, and the package default is `hbar = 0.658…`, so the
  manual scipy and the manual zand agreed with *each other* in a *different
  physical system* than the suite's. Any number produced outside the suite
  environment is not comparable with suite numbers. See §6.
* `zand − nozand` at machine precision did not flag any of this: both MPI
  codes read the same `eps` and share the integrator code, so they are not
  independent checks of each other's integration error — the general-purpose
  solver is. That lesson stands.

**But the eval grid costs time, and it also costs memory.** Every extra eval
point is another `solve_ivp`, and the peak memory scales with that count —
`SR` at a fixed `t = 0 … 2` needs >12 GB at N = 50 but only 3.5 GB at N = 10
(see §1); `scipy_vs_ARK4_Zand` behaves the same way. The window matters too, but
N is the cheap lever. The shipped defaults are therefore modest, and
`ZP_CHUNK` is the structural fix: it integrates several eval points per solver
call, so the same grid costs a fraction of the memory (measured on `SR`:
3658 → 1657 MB, see §7 item 7).

The solver settings themselves matter much less:

* `scipy_ode` `atol`/`rtol`: on `synth_1` the metric is identical (4.4–4.5e-06)
  from `1e-6/1e-4` down to `1e-12/1e-12`, and on `scipy_vs_ARK4_Zand` tightening
  `1e-6/1e-4 → 1e-8/1e-8` left `max|dJ|` at 2.855. The exception is `SR`, where
  the original `rtol = 1e-3` was loose enough to get the occupation number wrong
  (`N_end` 5.7562 vs 5.8406); at `1e-8` it matches to 4e-4, so `SR` uses
  `1e-8/1e-8`. This is why `--fast` can loosen the tolerances without changing
  the picture.
* MPI side: `eps` 1e-10 → 1e-8 moved `synth_1` by ~6% (5.06e-04 → 4.78e-04);
  `stepsize` 0.1 → 0.05 changed nothing, the adaptive controller overrides it.

### 3.4 The two Python propagators are set up differently

`scipy_ode` and `AdaptiveRK4` are not two implementations of one scheme. They
share the start state and an equivalent right-hand side (the `scipy_ode` path
uses shortlib's `f` by default, `AdaptiveRK4` the package's — the two agree to
~1e-15, §7 item 8), and almost nothing else — worth knowing, because a metric
that looks bad is often the *setup*, not the solver.

| | `AdaptiveRK4` | `scipy_ode` |
|---|---|---|
| signature | `(f, sig0, psi0, omega0, eps, t0, t1, dH, delta, Ixi, h_guess, …)` | `(f, sig0, psi0, omg0, t0, t_eval, dH, delta, Ixi, …)` |
| time argument | a closed interval `t0 … t1` | an **array** of output times, `t_eval` |
| integration | one sweep on an internal adaptive grid | **one `solve_ivp` per `t_eval` point**, chained |
| integrator | hand-written RK-Fehlberg 4(5) | `scipy.integrate.solve_ivp`, default method |
| tolerance | `eps`, the 5th positional | `atol` / `rtol` |
| first step | `h_guess` (else `(t1-t0)/1000`) | none (`dt_guess` accepted but unused) |
| returns | `(t, dict)`: `current_left`, `current_right`, `density matrix` | `(t_eval, rho, cl)`: DM of k=0 and the **left current only** |
| writes | `<name>.txt`, `<name>_last_*.npy`, `_times`, `_current_*`, `_#electrons_device` | nothing |

Three consequences that bite:

* **The output grids are not comparable.** `AdaptiveRK4` reports on its own
  adaptive grid (dense); `scipy_ode` reports only at its `t_eval` points.
  Interpolating the coarse one onto the dense one is what inflates `max|dJ|`
  (§3.3). On `Honeychurch2018` that is the whole story: 1.92e-02 at 200 points →
  1.33e-03 at 2000, against `AdaptiveRK4`'s 1.80e-03.
* **`eps` and `atol`/`rtol` are different knobs**, and in the tests they are not
  calibrated against each other (`Honeychurch2018` runs `eps = 1e-12` beside
  `atol = rtol = 1e-8`). `eps` bounds the *absolute* Frobenius norm of the 4(5)
  error estimate over the whole state (the `TERR` closure in
  `decoupled_functions.py`), which makes it much tighter in practice than its
  value suggests — and it is why ARK4 is slow.
* **`method=` on `scipy_ode` does nothing.** Its `propagate()` helper calls
  `solve_ivp` without forwarding `method`, so every `scipy_ode` run uses SciPy's
  default RK45 whatever the caller asks for; `dt_guess` is dead for the same
  reason. The MPI codes *do* honour `RK_method` from the environment, so a
  mismatch there is real, which is why every test exports `RK_method=RK45` (§5).

One thing that is *not* a difference: `AdaptiveRK4`'s estimator is only
degenerate (it returns `1` unconditionally) when `atol` is passed to it. No test
passes it, so they take the `sqrt`-form branch and the step adapts normally.

## 4. Files a test needs

| File | Purpose |
|---|---|
| `<device>.py` | builds the device, runs electrodes + TBtrans, fits the self energies, runs the Python propagator and writes the MPI input arrays via `Test.tofile(name, tol=…)` |
| `Initial.py` | run length/step/tolerances, `Arrs = load_dictionary(name+'/Arrays/')`, and the `ZP_T0`/`ZP_T1`/`ZP_SAVEEVERY` overrides |
| `shared.py` | `dH(t, σ)` and `delta(t, a)` — orthogonal basis, used by every method |
| `Bias.py` | `dH`, `bias`, `dissipator` for `zand` (re-exports `shared`) |
| `Bias_NO.py` | the same for `nozand`, with `dH` transformed to the nonorthogonal basis |
| `run.sh` | the pipeline above |
| `run_steady.sh` | optional *separate* pipeline: device → `SCF` + `psinought` → both propagators. Run through the driver's `--steady` mode; some tests have none. |
| `get_steady_state.sh` | the `SCF` + `psinought` step that `run_steady.sh` calls |
| `compare.py` | the figure plus the printed metrics (including `zand - nozand`) |
| `inventory.txt` | minimal list of the files a test consists of (everything not listed there is regenerated by `run.sh`) |

Every Python driver starts with the same block, which picks up the fast-mode
overrides and loads the start state:

```python
import os
T0 = float(os.environ.get('ZP_T0', <test's own t0>))   # fast mode
T1 = float(os.environ.get('ZP_T1', <test's own t1>))
N  = int(os.environ.get('ZP_NEVAL', 200))
ATOL = float(os.environ.get('ZP_ATOL', <test's own>))  # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', <test's own>))
...
C.get_dense_matrices(tol)
from Zandpack.Loader import flexload
A       = 'NAME/Arrays/'
C.sigma = flexload(A + 'DM_Ortho.npy')      # the same files the MPI codes read
C.xi    = flexload(A + 'xi.npz')
C.Ixi   = flexload(A + 'Ixi.npz')
C.get_propagation_quantities()              # rebuild psi/omega from xi/Ixi
...
from shortlib.driver import propagate       # ZP_RHS / ZP_CHUNK, see §5
t, dm, jl = propagate(C, dH, delta, T0, np.linspace(T0, T1, N) + 1e-5, ATOL, RTOL)
```

`Bias_NO.py` is the only difference between the two MPI runs:

```python
from Zandpack.Help import TDHelper
from shared import dH as dH_orthogonal          # dH is written orthogonally
Hlp = TDHelper('NAME'); iL = Hlp.invLowdin
def dH(t, sigma): return iL @ dH_orthogonal(t, sigma) @ iL
```

**Gotcha:** `nozand` imports the bias file *under the module name `Bias`*, so
`Bias_NO.py` must not `import Bias` (circular self-import). That is why `dH`
lives in `shared.py` and both bias files re-export it.

## 5. Conventions and gotchas

* **Same starting point.** `scipy_ode` and `zand`/`nozand` must start from the
  same `sigma`, `psi`, `omega`. This is guaranteed by construction: every Python
  driver loads those from the same files the MPI codes read (above), right after
  `get_dense_matrices()` — which is the call that *sets* `sigma`, so the
  assignment has to come after it; `get_propagation_quantities()` does not touch
  `sigma`, so it can be called again to rebuild `psi`/`omega`. Because both sides
  read the same arrays, this also picks up any refinement written into them —
  `SCF` rewrites `DM_Ortho.npy` (keeping the previous one as `oldDM_Ortho.npy`),
  which is exactly what made `scipy_vs_ARK4_Zand` look 2.9 when only the MPI side
  got the steady state (16.82 vs 16.00 electrons). Verified after the fix on
  `synth_1`'s steady-state pipeline: `override == DM_Ortho: True`, and the two
  sides still agree to `max|dJ| = 1.02e-06`.
* **Tolerances.** `tofile(tol=…)` and `get_dense_matrices(zero_tol=…)` are set to
  the same value in every test: 1e-8 (`Croy2009`, `Honeychurch2018`,
  `scipy_ode_vs_zand`, `SR`), 1e-12 (`PopescuCroy2016`), 1e-10
  (`scipy_ode_vs_AdaptiveRK4`), 1e-4 (`Rib`), 1e-5 (`scipy_vs_ARK4_Zand`), 1e-7
  (examples).
* **`RK_method=RK45`** is exported by every `run.sh`. The MPI codes read
  `RK_method` from the environment and fall back to `Zandpack.k0nfig.RK_Method`,
  whose default is `DOP54` — so without this line the MPI runs use a different
  integrator than the label suggests.
* **Save interval vs window.** The MPI codes write current/DM every `saveevery`
  substeps. If the window is shorter than that, the run produces an *empty* save
  directory and `N()`/`J()` have nothing to read (this broke `small_bound` in a
  0–2 window with `saveevery=50`). `--fast` therefore sets `ZP_SAVEEVERY=5`.
* **Always clear the save directory *and* the directory it is moved to.** The
  MPI codes *append* `current_<lead>_<n>.npy` / `DM<n>.npy` chunks — they do not
  overwrite the directory. If the directory a run writes into still holds chunks
  from an earlier run, `J()`/`N()` stitched the old and the new series together
  and the comparison was silently wrong. Every `run.sh` therefore clears **both**
  directories before the run (`rm -rf <name>_save <name>_save_zand`), not just
  the target before the `mv`. `synth_1` was hit by exactly this: a leftover
  `synth_save/` made its smooth `zand` current look 0.106 away from the others
  when it was really 2.2e-05 away. `run_sawtooth.sh` has to clear
  `synth_save*_saw` the same way, and it also restores `Bias.py` through a
  `trap`, so an interrupted run cannot leave the sawtooth drive swapped in for
  the next smooth run.
* **Stale results.** A failed Python driver leaves the previous
  `*_results.npz` in place, and `compare.py` will then report the *old* metric
  (this is how an earlier "small_bound is grid-independent" reading, and the
  `ok` status for `scipy_vs_ARK4_Zand`'s parked run, were wrong). Check the
  timestamp of the npz against the save directory.
* **`zand` vs `nozand` keywords.** `nozand` understands `bias_file=`,
  `initial_file=`, `out_dir=`; **`zand` takes only `Dir=…`**. To run `zand` with
  a different bias you must swap the file (`cp Bias_saw.py Bias.py`, restore
  afterwards) — see `synth_1/run_sawtooth.sh`. This is also why a sawtooth run
  can silently compare the wrong thing.
* **k-points.** Both `scipy_ode` and `J()` return one current column per
  k-point; the compare scripts sum them (`total()` helper) for the total current.
  `J()` *stitches* several directories together (it is meant for restarted runs),
  so pass exactly one save directory per comparison.
* **`D_lead_ortho`.** Use `read_data(..., D_lead_ortho=False)` when the
  TBtrans/Siesta call really samples k-points (e.g. `kp_tbtrans=[1,5,1]`); keep
  the default `True` for `nk = 1` (including a spin-polarised `nk = 2`, two
  copies at Γ). `scipy_ode_vs_zand` and `scipy_vs_ARK4_Zand` are the k-sampled
  ones.

  The rule exists because `D_lead_ortho=True` computes the Yan Ho Kwok et al.
  (2013) lead–device orthogonalization correction self-energies (`Sig0`/`Sig1`,
  `TimedependentTransport.py` ~line 449) and feeds them into `Sys.iG()`, which
  changes the Löwdin transform, the Gammas, the fitted self-energies and the
  equilibrium start DM. The old bug on that path (k-sampled systems) has been
  **fixed in the package as of 2026-10-24**; the fix also changes the modelled
  physics slightly, so post-fix `True` runs are *not* expected to reproduce
  pre-fix particle numbers. Both tests carry a `ZP_DLO=1` env override in their
  device scripts to run the `True` variant without editing anything.

  Post-fix A/B (`ZP_DLO=1`, 2026-10-24, `scipy_ode_vs_zand` t = 0…20,
  `scipy_vs_ARK4_Zand` t = 0…1): `True` runs cleanly on both, no crash.
  `scipy_vs_ARK4_Zand` is unchanged within its sampling noise. On
  `scipy_ode_vs_zand` the MPI codes still agree with each other (zand − nozand
  2.1e-05) and produce N_end(k=0) = 5.974 — but the *converged* Python
  trajectory (tight tolerances `1e-10/1e-10`, or a dense N = 1000 grid) lands
  on N_end = 6.0054 with max|dJ| ≈ 0.54, whereas under `False` the converged
  Python trajectory agrees with the MPI codes to 6.9e-04. The loose N = 200
  `True` run agreeing at 8.3e-02 is an under-resolved coincidence, not
  evidence of agreement. **Reading: with the `Sig0`/`Sig1` corrections present,
  the Python side (`make_f`/shortlib + its start-state reconstruction from
  `Arrays/`) and the MPI codes solve slightly different initial-value
  problems.** The corrections enter `Arrays/` in a form the MPI codes act on
  and the Python reconstruction does not fully reproduce (or vice versa) —
  `zand − nozand` cannot see it because both read the same arrays. This is the
  first thing to check if `D_lead_ortho=True` is ever wanted for k-sampled
  runs; it lives in the package, which is read-only from here.
* **Dated figures.** Every `compare.py` (and `SR/check.py`) saves its figure
  twice: under the canonical name (overwritten each run — what the docs
  reference) and as `<name>_<YYYY-MM-DD_HH-MM-SS>.png`, which is kept, so each
  run's figure survives for the regression history alongside the timestamped
  logs in `logs/`.
* **Sparse scipy_ode output (SR).** `scipy_ode`'s memory grows with the number
  of eval points, and its peak also scales with restarts. When only the current
  and occupation at a few well-chosen times are wanted, set `ZP_TLIST`
  (e.g. `ZP_TLIST=0.1,0.5,1,2,5,10,15,20 sh ...`): the ODE is still integrated
  adaptively through the whole window, but output is produced only at those
  times, in one `solve_ivp` call with `ZP_CHUNK` covering them all — measured
  on SR: **1.1 GB** instead of 6–9 GB. Conventions on a sparse grid (directive):
  draw the sparse points as **scatter crosses** (`marker='x'`), never a line
  through few points; and report **max|dJ| only**, computed *pointwise at the
  sample times* (the dense curve sampled there) — never an rms of a coarse
  curve interpolated onto a fine grid, which manufactures differences the
  methods do not have. `SR/check.py` implements both (`sparse` branch).
* **A `dH` that reads `sigma` needs *both* Löwdin transforms in `Bias_NO.py`.**
  `nozand` hands `dH` the density matrix in the **nonorthogonal** basis, while
  every test writes `dH` in the orthogonal one. So the nozand bias file has to
  convert the argument as well as the result:

  ```python
  def dH(t, sigma_NO):
      sigma_O = iL @ sigma_NO @ iL          # iL = Hlp.invLowdin
      return iL @ dH_orthogonal(t, sigma_O) @ iL
  ```

  Every other test's `dH` ignores `sigma`, so their `Bias_NO.py` only transforms
  the result and this cannot show up. `synth_dm/` is the test that exercises the
  **transform** deliberately: it is the only non-orthogonal device with a
  sigma-dependent `dH` (`S != I`, `max|L − I| = 0.11`; with `S = I` one has
  `iL = I` and the argument transform degenerates to a no-op). **SR's `dH` also
  reads `sigma`** (a mean-field term in `shared.py`), but its device is
  orthogonal (`S = I`), so its single-transform `Bias_NO.py` is correct there.
  (Getting the direction from the
  package: nozand builds its start state as `sig0 = Sm05 @ DM_Ortho @ Sm05`
  with `Sm05 = Hlp.Lowdin`, i.e. `dm_NO = L dm_O L`, so `dm_O = iL dm_NO iL`.)
* **Periodic device couplings.** When the device Hamiltonian is built by hand
  from a geometry that carries `nsc` along the transport direction, the ends stay
  coupled to their periodic images. Call `dev_H.set_nsc((1,1,1))` before
  `construct()` — without it the MPI and Python propagation disagree badly (in
  `scipy_ode_vs_AdaptiveRK4` the cross-method `max|dJ|` drops from 5.0e-01 to
  1.2e-03) and in `scipy_vs_ARK4_Zand` the MPI propagation diverged outright.
* **Currents from the RK4 runs are 2-D** `(n, 1)` when read back from the
  `*_results.npz`; `.ravel()` them before feeding `metrics()`, otherwise the
  subtraction broadcasts silently into a wrong number (or raises).
* **`AdaptiveRK4` is slow.** Its `eps` bounds the *absolute* Frobenius norm of
  the 4(5) error estimate over the whole state, and every accepted step runs an
  inner `while TE > eps:` refinement loop, each attempt costing six RHS
  evaluations. That is why `scipy_vs_ARK4_Zand` was ~3 s/step there, and why its
  Python side is `scipy_ode`. (Its estimator is degenerate — returns `1`
  unconditionally — only when `atol` is passed, which no test does; see §3.4.)
* **The Python right-hand side is a choice now (`ZP_RHS`).** Every test's Python
  driver propagates through `shortlib.driver.propagate(...)`, which picks the
  right-hand side from the environment: `shortlib` (the default) is the
  re-implementation in `shortlib/rhs.py`, and `ZP_RHS=make_f` switches back to
  the package's `TD_Transport.make_f`. Verified equivalent to ~1e-15 on three
  devices (`shortlib/verify_rhs.py`), and across the whole suite the fast run's
  SUMMARY table comes out identical to the `make_f` one while finishing ~10 %
  faster (403.0 vs 446.8 s fast, 1520 vs 1745 s full, ten-test runs). The
  per-call ratios are 1.55x / 2.43x / 2.50x on `synth_dm` / `SR` /
  `scipy_ode_vs_zand`. `ZP_CHUNK=K` additionally integrates K eval points per
  `solve_ivp` call (the piecewise method, `shortlib/piecewise.py`), with
  `ZP_TOL_TIGHTEN` tightening the tolerances to compensate — that is what bounds
  the peak memory (`SR`: 3658 → 1657 MB). `shortlib/README.md` has the
  measurements.
* **`AdaptiveRK4_Opti`** only accepts the `purenp` data path
  (`get_dense_matrices_purenp()` + `make_f_purenp()`); with `make_f()` it raises
  an axis error. `purenp` is reported to be buggy, so the tests use `make_f()`.
* **`parallel_E`.** `SiP.Real_space_SI/SE(parallel_E=True)` goes through
  joblib's process pool and fails in this environment; the `Rib`/`SR` device
  scripts use `parallel_E=False`.
* **Memory.** The heavy Python sides must not run at the same time. Measured
  peaks: `examples/small_bound` ~11 GB at its full window with `make_f` (the
  largest), `SR` 6.2 GB at its own full window (N = 20) and 3.5 GB at N = 10,
  `scipy_vs_ARK4_Zand` 4.6 GB at its `t = 0…1` default. `ZP_CHUNK` cuts these
  substantially — SR is 3658 → 1657 MB at `t = 0…2`, N = 10.
* **Pickles and pp.** `TD_Transport.pickle()` defaults to lzma, so state is
  stored/loaded as `*.xz`. `pp_folder` defaults to `'../pp'`, so a `pp/`
  directory is expected next to the run directory: `TestRUN/pp` for the test
  directories, pointed at with `pp_folder='../../pp'` from `examples/`.
* **Steady state as a separate pipeline.** `SCF` + `psinought` rewrite
  `Arrays/`, so they must run *between* the device build and the propagators.
  Run it with the driver's `--steady` mode, which executes each test's
  `run_steady.sh` (`synth_1`, `examples/small_bound`, `scipy_vs_ARK4_Zand`);
  tests without one are reported and skipped. It is deliberately not part of the
  default `run.sh`.

## 6. Mistakes made in this pass

Recorded so they are not repeated — each one cost real time or produced a wrong
number at some point.

| mistake | how it showed up | lesson |
|---|---|---|
| Ignored the `D_lead_ortho` rule I had been given | `scipy_vs_ARK4_Zand` kept `read_data()`'s default while TBtrans was k-sampled; the user had to point it out | audit the rule across *all* `read_data()` calls, not the file under discussion |
| Wired `get_steady_state.sh` into the default `run.sh` | only the MPI side got the SCF density matrix (16.82 vs 16.00 electrons), which I then spent a long time diagnosing as a "method difference" | changing what a tool writes into `Arrays/` changes the start state of one side only — keep such steps as a separate pipeline |
| Passed `bias_file=Bias_saw.py` to `zand` | `zand` silently ignored it (it only takes `Dir=`), so the "sawtooth" run used the smooth bias and I reported `max|dJ| = 0.267` as a property of `zand` | verify that an argument is actually honoured before drawing conclusions from it |
| A `sed '0,/^plt.plot(T, JL/i …'` range insert | repeated the inserted block 38–42× in the three paper `compare.py` files, corrupting them | range-inserts are not portable; use `/pattern/i`, then check the insert count |
| Prepended `T0 = float(os.environ.get('ZP_T0', -3*np.pi/…))` above `import numpy` in `Honeychurch2018.py` | `NameError` at import; the test failed in the first fast-suite run | don't put expressions with library calls into a prepended header |
| `pkill -f "RK4.py"` | matched its own command line and killed my own shell | never `pkill` a pattern that appears in the command doing the killing |
| Drove the eval grids to 1000–2000 points for accuracy | OOM-killed `examples/small_bound` (1000 pts), `SR` (500 pts) and `scipy_vs_ARK4_Zand`; the user pointed out `scipy_ode` "eats memory" | sampling is a memory/accuracy trade; ship modest defaults and expose them (`ZP_NEVAL`) |
| Read a stale `*_results.npz` | reported "small_bound is grid-independent" when in fact its driver had failed and `compare.py` re-reported the old metric | check the npz timestamp against the save directory |
| `J(['dir_a', 'dir_b'])` in `sweep.py` | `J()` *stitches* directories (meant for restarted runs), so the first sweep compared against a concatenated series and produced meaningless numbers | pass one save directory per comparison |
| Plain `mv <name>_save <name>_save_zand` in every `run.sh` | re-running a test nested the new save dir inside the old one; found only because the fast suite re-ran tests | clear the target first |
| Built the fast mode without a save-interval override | with a 0–2 window and `saveevery=50` the MPI run wrote nothing, so `N()` had no data and `small_bound` failed | when shortening a window, shorten everything that samples it |
| Wrote `metrics()` without ravelling the RK4 currents | `Honeychurch2018` crashed; in `Croy2009`/`PopescuCroy2016` the same mistake *silently* broadcast a `(n,1)` against `(n,)` into a wrong number | check array shapes before trusting a difference metric |
| Added `MPLBACKEND` only to the scripts I had written | the first `RK_method` insertion silently skipped the three paper tests | verify the effect of a bulk edit with a grep, not by assuming the pattern matched |
| Cleared only the `mv` *target*, not the directory the MPI code writes into | the codes append chunks, so a leftover `synth_save/` was stitched into `synth_save_zand`; `synth_1`'s smooth `zand` current read 0.106 away from `nozand` when it was really 2.2e-05 | clear the source *and* the target before every MPI run — the append-vs-overwrite behaviour of the code decides which |
| Left `run_sawtooth.sh` without the `rm -rf` guards at all, and without restoring `Bias.py` on failure | `synth_save_zand_saw/` grew a nested `synth_save/`, so `J()` stitched a full-window run into a fast one; an interrupted run could leave the sawtooth drive swapped in for the next smooth run | a file-swap that a program cannot be told about needs a `trap` to undo it |
| Called `logs/run_fast_2026-10-16_10-34-26.log` the baseline without reading it | it actually had `Honeychurch2018` failing (`rc=1`, a traceback in `compare.py`), while §3 listed metrics for it — from an older run | read the baseline before quoting it; a test that says `ok` in prose but `rc=1` in the log means the doc is stale |
| Took the handoff's "`nproc` reports 1" at face value | designed the fast mode around a single core | `nproc` honours `OMP_NUM_THREADS`, which is exported as 1; the box has 18 CPUs (`/proc/cpuinfo`, `os.cpu_count()`) |
| Believed "`scipy_vs_ARK4_Zand` completes in 125 s at `ZP_NEVAL=20`" | re-running it grew past 22 GB at 20 points and had to be killed; the note was stale | measure a claim before relying on it — and the memory tracks the **eval-point count** (`SR` at a fixed window: N = 50 → >12 GB, N = 10 → 3.5 GB), so N is the lever |
| Assigned a flat `(n_mode, no)` result to `D_psi[k]`, which is 4-D | with `parallel=True` numba *compiled it anyway* and returned numbers 0.589 off the reference — no exception | a shape-mismatched assignment can mis-compile silently under parfors; verify against a reference, and check the shapes |
| Trusted a probe that said numba supports `np.matvec`/`np.vecdot` | it passed arrays as function *arguments*; in `make_f`'s closure the same calls are rejected as "unsupported", so the first compiled attempt failed | reproduce the real call context (closure vs argument) before building on a capability test |
| Called the second `D_lead_ortho=True` run a "re-run after the fix" | both runs postdated the package fix (00:16) by hours, so "identical results" meant "same code", not "the fix changed nothing" — I had compared two post-fix runs and never measured the pre-fix behaviour | before interpreting an A/B as before/after, check the mtime of the thing that was supposedly changed |
| Wrote that SR's 1.04 was "interpolation artefact" without measuring it | a densified run *seemed* to disprove it (metric grew), which triggered a wrong eps-based explanation | an artefact claim needs the artefact experiment before it goes in a document — but see the next row: the densified run was itself invalid |
| Ran manual comparison experiments without the suite environment | the manual scipy and manual zand both ran with the package default `hbar = 0.658…` instead of the suite's `ZANDPACK_HBAR=1.0`, so they agreed with *each other* (1.2e-03) in a different physical system; the resulting "eps is the cause" story looked airtight and was wrong — the real within-suite eps effect is 0.059, and SR's true pointwise agreement is 4.0e-04 | every manual run must export the same env block the suite uses (`ZANDPACK_HBAR`, `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RK_method`) — a shared env var that changes the physics silently manufactures method differences |
| Trusted `zand − nozand` at machine precision as evidence of MPI-side accuracy | both codes read the same `eps` and share the integrator code, so their errors are correlated, not independent | the independent check on the MPI integrator is the general-purpose solver (scipy_ode), not the other MPI code |

## 7. Open items (start here)

The regression check for any change is `./run_zandpack_tests --fast` (~9 min)
diffed against `logs/run_fast_2026-10-17_08-38-42.log`; before a handoff, once
more with the full windows, diffed against `logs/run_2026-10-17_08-51-53.log`.

Nothing here is blocking: every test runs, prints its metrics and is in the
driver. What is left is fidelity and coverage.

1. **Done — both modes now have a reference log at the shipped settings.**
   `logs/run_fast_2026-10-17_08-38-42.log` (fast) and
   `logs/run_2026-10-17_08-51-53.log` (full, 2032 s with `ZP_JOBS=1`) each cover
   all eleven tests *after* the save-directory fix, so §3 is backed by runs that
   can be diffed. The one thing still open here is fidelity, not the harness:
   the shipped Python grid is N = 200, so the full-window numbers are
   sampling-limited (§3.3), and tightening them costs memory (item 5).
2. **~~`SR` is not in the driver's test list.~~ Done.** `SR` is in `TESTS` and in
   `HEAVY`, so it runs on its own; in fast mode it gets
   `ZP_T1=2, ZP_NEVAL=10, ZP_CHUNK=10` (its `check.py` name collision is fixed
   and its `ZP_NEVAL` default is 20 for the full window). Its Python side is
   memory-hungry, which is why it is heavy — see §5 for the peak-RSS numbers and
   `shortlib/README.md` for what chunking does to them.
3. **Steady-state mode covers 3 of the 12 tests.** The `--steady` mode works
   (`synth_1`, `examples/small_bound`, `scipy_vs_ARK4_Zand`); the remaining
   tests still need a `get_steady_state.sh` + `run_steady.sh` pair, which is
   mechanical — the Python-side half (reading `sigma`/`xi`/`Ixi` back from
   `Arrays/`) is already in every driver. `Honeychurch2018` now writes its
   arrays before it propagates, so it is ready for one. `synth_dm` is the one
   where a steady state would matter most, since a mean-field feedback has a
   self-consistent ground state — but the SCF would then have to include the
   feedback too, so it is a real piece of work rather than a copy of the
   template. `ThesisZGNR`'s steady-state preparation exists as
   `examples/ThesisExample/TDeq_1.sh` (ts-contour → `modify_occupations` →
   `SCF` → `psinought`) and mainly needs to be wrapped in the two scripts; a
   `ThesisZGNR` steady run would also be the natural place to see whether the
   non-PSD fit's worries shrink when the start state is self-consistent.
4. **The synthetic bench can be extended — but the obvious next step is
   blocked.** `synth_1` has one orbital per site, no k-point sampling and a
   single transverse cell. A companion bench with a two-site-wide, a2-periodic
   ladder (so TBtrans really samples k) was prototyped: the device and TBtrans
   run, but `read_data(D_lead_ortho=False)` then dies *inside the package* with

   ```
   Gf_Module/Gf.py:801  ValueError: non-broadcastable output operand with
   shape (49,1,2) doesn't match the broadcast shape (49,2,2)
   ```

   — a multi-orbital transverse cell combined with real k sampling is not
   supported on this path. A workaround has to live in the test (or the k-path
   has to go through `scipy_ode_vs_zand`-style SIESTA electrodes instead of
   `manual_H`); do not patch `Gf.py`. Because of that, the cheap bench cannot
   reach the `D_lead_ortho=False` / `total()` paths — those are only covered by
   the two real k-sampled tests.
5. **The full-window Python grid is memory-bound.** Only the fast mode is
   comfortably under the ceiling; raising `ZP_NEVAL` in a full run trips the
   `scipy_ode` growth (it scales with the eval-point count). `ZP_CHUNK` is the
   way to have both, but it is deliberately not the default for the full runs so
   that their numbers stay comparable with the fast ones.
6. **`examples/` still holds many untested example directories** (`AGNR+Tip`,
   `AuBreakJunction`, `C60`, `GroH`, `GrRibGr`, `Hubbard`, `Pohl`, `Pohl2`,
   `TB_ribbon`, `Testing`, `VerysmallRSSE`, `ZGNR+N`, `ThesisExample`, `misc`,
   `generic_wrapper`, `nb_tutorial_*`). `examples/AGENTS.md` asks for more
   nozand-vs-scipy_ode tests; keep any new one below ~50 orbitals (`Rib`'s 694
   are excluded on purpose).
7. **Piecewise `scipy_ode` — the structural fix for the memory problem.**
   `scipy_ode` holds the whole ODE state and its peak grows with the number of
   eval points (and drifts up during a run), which is what makes `SR` and
   `scipy_vs_ARK4_Zand` need `HEAVY` and a small grid. The state is
   `(Sigma, Psi, Omega)`; if a propagation is done in short pieces, keeping only
   the current from each piece and restarting from the returned state, the peak
   stays bounded regardless of the window. **Caveat:** the package's
   `scipy_ode` does **not** currently return the state — line 697 of
   `decoupled_functions.py` is
   `return t_eval, np.array(rho), np.array(cl), #state_sig, state_psi, state_omg`,
   i.e. `Psi` and `Omega` are commented out (only `Sigma` is recoverable, as
   `rho[-1]`). So this has to be a **test-side** segment driver over
   `solve_ivp` using the package's `f` (plus `one2three`/`three2one` and
   `PI`/`Ixi` for the current), not a chain of `scipy_ode` calls.

   **Done: `shortlib/piecewise.py`.** `solve_pieces(..., chunk=K)` integrates K
   eval points per `solve_ivp` call, carries the state across pieces and keeps
   only the current; it uses the package's own
   `one2three`/`three2one`/`PI`/`Jk`, so only the loop is re-implemented.
   `chunk=1` reproduces the package's `scipy_ode` **bit for bit** (max|dJ| =
   0.0 on `synth_1`), which is the check that the segment loop is right.
   `shortlib/compare_scipy_ode.py` runs the four combinations, each in its own
   process, and diffs them. Measured on `SR` (`t = 0…2`, N = 10): peak RSS
   3658 MB (`scipy_ode` + `make_f`) → 1657 MB (`solve_pieces` chunk=10 +
   `make_f`) → 1564 MB with shortlib's `f`, while the left current moves by
   5.7e-06 — the solver restart, not a transcription error, and far below the
   ~1e-4 metrics the tests print. Shortlib's `f` reproduces the tests' results
   to 2e-15 on both `synth_1` and `SR`. **Wired into every test**: the choice
   lives in `shortlib/driver.py` (`ZP_RHS`, `ZP_CHUNK`, `ZP_TOL_TIGHTEN`) and
   all twelve drivers call it, so there is no per-test plumbing to keep in sync.
   The defaults are `ZP_RHS=shortlib` and `ZP_CHUNK=1`; `SR` additionally gets
   `ZP_CHUNK=10` in fast mode. Run inside `synth_1` (N = 200, t = 0…20):
   `shortlib` + chunk=1 agrees with the package path to `max|dJ_l| = 1.5e-16`,
   and chunk=10 to 1.6e-07 (relative ~1e-6 of the current).
8. **A short library replicating `make_f` (and its siblings).** Wanted: a small
   library in this folder that rebuilds what `make_f`/`get_dense_matrices`/
   `get_propagation_quantities` construct, (a) verified to agree with `make_f`
   numerically, (b) faster, and (c) able to take a different input than the
   package's own pickled object — e.g. the arrays as written to `SRC/Arrays/`.
   That would let the tests bypass the `purenp` path that is reported buggy and
   give the piecewise driver above a clean, controllable state. It is also the
   natural home for the segment loop of item 7.

   **Started: `shortlib/` — the right-hand side is done, verified, and is now
   what the tests use.** `shortlib/rhs.py` re-implements `make_f` from the
   equations in `Zandpack.equations` as a numba function with the same
   signature; `smoke_f.py` (seconds) and `verify_rhs.py` (minutes) check it
   against `make_f` and time it, both exiting non-zero on mismatch. Verified
   equivalent (~1e-15 relative) on `synth_dm`, `SR` and `scipy_ode_vs_zand`, on
   the real propagation state and on randomised states, at two `dH`s and with
   `hbar != 1`. Speed-up per RHS call with the shipped default
   (`dense_omega=False`): **1.55x / 2.43x / 2.50x**; the `dense_omega=True`
   variant is 3.1x on the tiny device but 0.5x on the 7.5 %-full map (it
   densifies Omega to `n_mode**2`), so it stays selectable but not default.
   Still to do: the *construction* half (`get_dense_matrices`/
   `get_propagation_quantities` and reading `<name>/Arrays/` directly).

   **Reviewed for headroom with `profile_rhs.py`** (which switches blocks off by
   doctoring the index maps, so it times the same arithmetic): the
   auxiliary-mode block is **93-98 %** of the RHS on the devices that matter,
   and it is now memory-bandwidth bound — on SR it gathers ~640 MB per call
   (~12 GB/s). Two cheap wins were taken during that review (hoist `conj(psi)`
   out of the row loop, precompute `chi_m` as a vector), moving SR
   70.7 → 53.9 ms and `scipy_ode_vs_zand` 8.9 → 7.9 ms with results unchanged to
   1e-15. The only remaining structural idea is a row-blocked gemm form of the
   two per-entry dots (and a blocked `Omega @ xi`); it should roughly halve the
   omega block at 36 % fill but not at 7.5 %, so it was left out. Measured
   breakdown and the reasoning are in `shortlib/README.md`.

   Gotcha found while doing it, worth not rediscovering: **numba accepts
   `np.matvec` / `np.vecmat` / `np.vecdot` only when the arrays are function
   arguments**; on a closed-over array (or module global) it raises "Use of
   unsupported NumPy function". `f`'s signature is fixed by
   `scipy_ode`/`AdaptiveRK4` and all its arrays are closed over, so only
   `@` / `np.matmul` can be used. Their conventions also differ:
   `np.vecdot(a, b) = sum(conj(a) * b)` conjugates the *first* argument (and
   `np.vecmat` does too, so it is not `a @ B`), while `np.matvec(A, v) = A @ v`
   conjugates nothing, and `np.vdot` flattens its arguments.

## 8. Changes made in this pass

### ThesisZGNR — a real SIESTA device with a mean-field dH (2026-10-24)

* **New test `ThesisZGNR/`**, adapted from `examples/ThesisExample` (5a/5b/5d):
  a 3-ZGNR built by SIESTA (SZ basis), TranSIESTA + TBtrans, pz-only device
  orbitals (69), and a **sigma-dependent `dH`** (mean-field `Rep = 2` on two
  mid-ribbon sites) with a box-pulse bias. The fit uses **25 Lorentzians per
  lead** where the thesis used 81, to keep the auxiliary-mode count at test
  size. The steady-state preparation (TDeq_1.sh: contour, modify_occupations,
  SCF, psinought) is deliberately *not* in the default run — same policy as
  everywhere else; it belongs in a future `run_steady.sh`.
* **Electrode TSHS fix (same family as the device `set_nsc` fix).** TBtrans
  aborted with "Electrode connectivity is not perfect" until the electrode
  Hamiltonians were re-read and re-written with the full coupling range:
  `Hm = E.read_TSHS(); Hm.set_nsc((3,1,1)); E.manual_H(Hm)` after the
  electrode SIESTA runs and before the device run. Recorded because the next
  SIESTA-based test will hit the same wall.
* **numba-compatible `shared.py`.** The shortlib RHS calls `dH`/`bias` from
  compiled code, so they must be `@njit`: the box pulse is inlined (the
  package's `box_pulse` is not jitted), and the drive term is precomputed as
  `M = L @ diag(Rmp) @ L` (the drive is linear in the bias, so
  `L @ diag(Rmp*V) @ L = V*M`) — no matrix products at call time. The direct
  form fails numba's typing on the closed-over `L` global.
* **Memory settings specific to this test.** At `atol = rtol = 1e-8` scipy's
  RK45 takes ~1e-6-sized steps (fast fitted Lorentzian modes, max Γ eigenvalue
  ~265) and the package's `scipy_ode` path (chunk = 1) drifts past 30 GB — its
  memory grows with the window on this device, measurably 3 GB @ t = 2,
  18 GB @ t = 10. Defaults are therefore `atol/rtol = 1e-6/1e-4` and
  `ZP_CHUNK = 10` (piecewise, set inside `run_scipy_ode.py`), which cap the
  peak at ~5 GB. Pointwise agreement on `t = 0..2` at these settings: 2.1e-04.
* **First results** (t = 0..20, N = 200): scipy_ode − zand/nozand 8.4e-02 /
  1.1e-02; **zand − nozand 2.5e-04** — not machine precision, like synth_dm
  (2.3e-05), and 10× larger here. Both are the nonorthogonal devices; this is
  consistent with the documented nonorthogonal residual and scales with how
  far S is from identity (hand-built chain vs real SIESTA SZ). `N_end`:
  22.364 / 22.363 / 22.351. Whether the residual is the double Löwdin
  transform or something deeper in the nonorthogonal path is the standing
  question from synth_dm, now with a second data point.
* **Fit-quality warning (documented, not fixed):** the 25-pole fit has
  Γ eigenvalues down to ~ −13 — the same "extra care" situation as the
  original 81-pole ThesisExample, so it is not caused by the reduced pole
  count. It does not affect the test's purpose (all methods consume the same
  fitted arrays, so the cross-code agreement stands), but the currents are not
  physical predictions, and the negative eigenvalue is (part of) why the
  Python side needs the loose tolerances. The warning is printed by
  `device.py` after the fit and spelled out in `ThesisZGNR/README.md`.

### The SR discrepancy resolved (2026-10-24)

* **SR is clean.** Pointwise at well-chosen sample times and in the suite
  environment, scipy_ode agrees with `zand`/`nozand` to max|dJ| = 4.0e-04;
  the suite's printed 1.04 is the coarse interpolated grid, not the methods.
  Full story and numbers in §3.3; two wrong intermediate claims are in §6,
  including the environment lesson (manual runs must export the suite's env
  block — the package's default `hbar = 0.658…` silently defines a different
  system than `ZANDPACK_HBAR=1.0`).
* `SR/Initial.py` `eps = 1e-5 → 1e-8`: a real but small convergence
  improvement (moves zand by max 0.059); costs roughly 2× suite time on SR
  (467 → 913 s, now the slowest test). Kept; revert if the cost matters more
  than the last digit.
* `SR/check.py` gained: the npz save every other test has
  (`scipy_ode_results.npz`), a `ZP_TLIST` mode — and a sparse default list of
  well-chosen output times (memory 1.1 GB vs 6–9 GB; see §5) with pointwise
  max-only metrics and scatter-cross plotting on the sparse branch.

### Concurrency, fast mode and the save-directory fix

* **The Python right-hand side is now `shortlib`'s** (see §7 item 8). Every
  test's Python driver calls `shortlib.driver.propagate(...)`, which selects the
  right-hand side from `ZP_RHS` (`shortlib` — the default — or `make_f` to
  switch back) and can chunk the solver calls (`ZP_CHUNK`, `ZP_TOL_TIGHTEN`) to
  bound peak memory. The direct `scipy_ode(...)` calls in the test drivers are
  gone. On the ten-test suite, `ZP_RHS=shortlib` produced a SUMMARY table
  identical to the `make_f` one and finished sooner (403.0 vs 446.8 s fast,
  1520 vs 1745 s full) — that comparison is what justified the default; §7
  item 8 has the per-call numbers and `shortlib/README.md` the measurements.

* `run_zandpack_tests` now runs the tests **concurrently** (`ZP_JOBS`, default 3)
  instead of one after another, with the memory-heavy tests (`scipy_vs_ARK4_Zand`
  and `SR`) running on their own. The log is still written in the declared test
  order (a block is buffered until the tests before it finish), so two logs diff
  cleanly; live progress goes to stdout only.
* Fast mode finishes in **~9 min (551 s, 11 tests)** instead of ~16 min: `ZP_NEVAL`
  100 → 50 (fewer, cheaper `scipy_ode` segments) plus the concurrency above.
* `scipy_vs_ARK4_Zand` is **un-parked**: `FAST_BY_TEST` gives it `t = 0 … 1`
  with 10 points, which keeps its `scipy_ode` peak at 4.6 GB instead of blowing
  past 20 GB. (An earlier note here blamed the window alone; the controlled
  measurement on `SR` shows the eval-point count is the dominant lever — see
  §1.)
* **`SR` fixed and running.** Its `check.py` had a name collision — `N` (the
  eval-point count) was rebound by `from Zandpack.plot import …, N`, so
  `np.linspace(T0, T1, N)` raised `TypeError: 'function' object is not
  callable` and the test died in seconds. That, not memory, is why it had been
  left out of the driver. Its `read_data(D_lead_ortho=False)` was set as
  requested, and its `ZP_NEVAL` default is now 20 (memory scales with N; the
  200-point default needed ~20 GB). It now completes: peak 6.2 GB, `zand −
  nozand` = 2.2e-13, `N_end` 5.841 / 5.8433. Its `scipy_ode` metric (1.04) is
  interpolation artefact at Δt = 1, not a solver disagreement.
* **Fixed a silent corruption in every test.** The MPI codes *append* chunk
  files, so every `run.sh` now clears the save directory the code writes into
  *and* the one it is moved to before running
  (`rm -rf <name>_save <name>_save_x`). `synth_1`'s smooth `zand` comparison had
  been reading a stitched old+new current (0.106 instead of 2.2e-05).
* **§3's full-window table was regenerated** from a single complete run
  (`logs/run_2026-10-17_08-51-53.log`, all eleven tests, 2032 s serial). That
  settled a long-standing trap: the old rows were a *mixture of grids* (some at
  N = 1000–2000), so they were not comparable with each other. The new table is
  all at the shipped grid and now includes full-window rows for the three paper
  tests, which previously had none. At the shipped grid the full windows turn
  out to be affordable — Popescu's `t = 0…200` is 227 s, not hours.
* `synth_1/run_sawtooth.sh` got the same guards plus a `trap` that restores
  `Bias.py`, so an interrupted run cannot leave the sawtooth drive swapped in
  for the next smooth run; `run.sh` also recovers a leftover backup.
* **`Honeychurch2018` gained a `scipy_ode` run.** It used to have only
  AdaptiveRK4 on the Python side, and its cross-method difference stood out
  (~1.8e-03 vs ~1e-04 or better elsewhere). `scipy_ode.py` is not a separate
  file here: the scipy run sits next to the ARK4 call in `Honeychurch2018.py` so
  both get the same start state (and an equivalent right-hand side), and
  `compare.py` now prints all three pairs. Result: `scipy_ode − zand` = 2.4e-04
  against `AdaptiveRK4 − zand` =
  1.4e-03 (fast), i.e. the Python ARK4 is the outlier. Its window is 15× longer
  than any other test's, so its `scipy_ode` output grid default went 200 → 2000
  (max|dJ| 1.9e-02 → 1.3e-03); its start state now also comes from
  `TDT/Arrays/` like every other test.
* **New test `synth_dm`: a `dH` that reads the density matrix.** It uses the
  `out[0,i,i] = Rep * sigma[0,j,j]` mean-field pattern from
  `examples/ThesisExample/Bias.py`, so the equations of motion are **non-linear**
  — nothing else in the suite covered that. It is built on `synth_1`'s chain but
  is deliberately **non-orthogonal** (`s_hop = 0.2`, so `S != I`, verified as
  `max|L − I| = 0.11`), because on an orthogonal device the nozand argument
  transform is a no-op and cannot be checked. Its `Bias_NO.py` is therefore the
  only one in the suite that converts *both* the density matrix and the result.
  Validated: all four methods agree (`scipy_ode − zand` = 1.2e-04, `scipy_ode −
  nozand` = 1.2e-04). Control experiment: switching the feedback off (`Rep = 0`)
  leaves `zand − nozand` at 2.3e-05, so that residual belongs to the
  nonorthogonal path, not to the feedback.
* **`--steady` is now a driver mode.** `./run_zandpack_tests --steady` runs each
  test's `run_steady.sh`, so the SCF + psinought pipeline is a first-class
  separate test rather than a script hidden in one folder. `run_steady.sh` +
  `get_steady_state.sh` exist for `synth_1` (pre-existing), `examples/small_bound`
  and `scipy_vs_ARK4_Zand`; tests without one are reported and skipped.
  Verified: `small_bound` `zand − nozand` = 1.1e-14, `synth_1` = 9.0e-15.
* **§3.4 added**: a side-by-side of how `scipy_ode` and `AdaptiveRK4` are set up
  (time argument, integrator, tolerance knob, outputs), including the finding
  that `scipy_ode`'s `method=` argument is never forwarded to `solve_ivp`.
* **`shortlib/rhs.py` reviewed for headroom** with the new `profile_rhs.py`:
  two hoists taken out of the auxiliary-mode row loop (`conj(psi)` once per k
  instead of once per row, and `chi_m` precomputed as a vector), moving SR
  70.7 → 53.9 ms and `scipy_ode_vs_zand` 8.9 → 7.9 ms with the results unchanged
  to 1e-15. The full breakdown, the bandwidth argument and the one remaining
  structural idea are in §7 item 8 and `shortlib/README.md`.
* `MPLBACKEND=Agg` added to the three paper tests' `run.sh` (they were missing
  it, so they were not runnable standalone).
* `handoff.md`: corrected the `nproc`/"1 CPU" claim (the box has 18 CPUs), the
  baseline-log claim (the previous "baseline" had `Honeychurch2018` failing
  while §3 quoted metrics from an older run), the `AdaptiveRK4`-"estimator
  returns 1" explanation, and the fast-mode numbers; added the save-directory
  convention and the new mistakes.

### Earlier pass

* `scipy_ode_vs_zand`: added the `zand`/`nozand` stages, `Bias_NO.py`, a
  comparison figure and the metrics; the zand window was aligned with the
  scipy_ode window (`t1 = 20`).
* `scipy_ode_vs_AdaptiveRK4`: modernised (Zandpack imports/API), writes the MPI
  inputs, runs zand + nozand, compares — and its device Hamiltonian now gets
  `set_nsc((1,1,1))`, which cut the cross-method `max|dJ|` from 5.0e-01 to
  1.2e-03.
* `scipy_vs_ARK4_Zand`: modernised, `Zand` → `zand`, added the missing `tofile`,
  aligned the name, replaced the slow ARK4 Python side with `scipy_ode`. Its MPI
  propagation used to diverge; the fix (in `make_device.py`) is
  `Hd.set_nsc((1,1,1))` plus `read_data(D_lead_ortho=False)` because TBtrans is
  called with real k-point sampling. A hard Gaussian bias spike was softened.
  Parked in fast mode (Python-side memory).
* `Rib`/`SR`: imports fixed, `bias = delta` + `dissipator` supplied (the MPI
  codes require all three), save names aligned, `shared.py`/`Bias_NO.py` added,
  `check.py` rewritten to use `scipy_ode` + metrics, `run.sh` added.
* `Honeychurch2018`: its `compare.py` was loading the wrong `.npz` and used a
  `TDT_TDT_…` save name; both fixed, plus the `.ravel()` fix.
* `examples/small_bound` and `examples/SuperCond`: turned into full
  scipy_ode-vs-nozand tests.
* All tests: start state read from the MPI input arrays, `tofile` tolerances
  aligned, metrics printed (including the `zand - nozand` row), the three paper
  tests given metrics, `RK_method=RK45` exported, the window/sampling/save
  interval made environment-overridable, and `rm -rf <target>` before each `mv`
  so tests can be re-run.
* `run_zandpack_tests`: is now the suite driver — per-test status/elapsed/metrics
  plus a summary table, a timestamped and flushed log, and a `--fast` mode.

## 9. Where this suite should live (recommendation)

Today the same material exists in four places and they have already drifted:

| location | size | state |
|---|---|---|
| `ZP_testing/TestRUN/` | 245 MB | **current** — 12 tests, the driver, `--fast`/`--steady`, `logs/` |
| `ZP_testing/tests/` | 19 MB | older partial copy (`Croy2009`, `Honeychurch2018`, `PopescuCroy2016`) |
| `ZP_testing/test1/` | 40 MB | older scratch (`adap_RK4*.py`, `Chain_*.npy`) |
| `…/Zandpack/Zandpack/testing/` | 380 KB | git-tracked scripts-only copy, inside the package; `synth_1/` is empty |

**Recommendation: move the suite to a top-level `tests/` in the Zandpack *source
repo*, i.e. a sibling of the importable package — and delete the other three.**

```
Zandpack/                  # repo root (has .git)
├── Zandpack/              # the importable package — keep it free of tests
├── tests/                 # <- the suite goes here
│   ├── run_zandpack_tests  # the driver stays the entry point
│   ├── README.md           # requirements: SIESTA/TBtrans/mpirun, LaTeX, ~30 GB
│   ├── conftest.py         # resolves pp path + env for every test
│   ├── Croy2009/ … synth_dm/
│   └── pp/                 # gitignored
├── pyproject.toml         # + [tool.pytest.ini_options] testpaths = ["tests"]
└── .gitignore             # + tests/**/Device/, tests/**/*_save*, tests/**/Arrays/
```

Why not `Zandpack/testing/` (its current in-package home):

* **Packaging.** It sits *inside* the importable package and only misses being
  installed implicitly — there is no `__init__.py` there, and `setup.py` says
  `packages=['Zandpack']`. Add an `__init__.py`, or switch to explicit package
  discovery, and it starts shipping, with 245 MB of DFT output and pickles
  behind it. Setuptools' flat-layout auto-discovery *excludes* a top-level
  `tests/`, so a sibling can't be shipped by accident.
* **Edit boundary.** The package's own `AGENTS.md` is explicit: "DO NOT modify
  or add any code yourself unless explicitly prompted" — the package tree is a
  review-only zone. A test suite is edited constantly; the two roles conflict.
* **Data vs scripts.** The suite is ~380 KB of scripts and ~245 MB of outputs.
  Only the scripts belong in git. The in-package copy is already the
  scripts-only split; it is just in the wrong place.

Why not stay in `ZP_testing/TestRUN/`:

* **No version control.** `TestRUN` is not in any git repo (`git rev-parse` stops
  at the filesystem boundary), so there is no history and no way to pin a
  metric change to a commit — which is exactly what the timestamped logs are
  for. A sibling `tests/` comes with `git clone` and `pip install -e .`, so code
  and tests move together.
* **Location coupling.** `pp_folder` defaults to `'../pp'`, which is what ties
  every test to being exactly one level under a `pp/` directory (and duplicates
  it: `ZP_testing/pp` *and* `TestRUN/pp`). Moving is the moment to resolve it
  from one place instead — e.g. a `ZANDPACK_PP` environment variable read by
  `conftest.py`/a small helper — and likewise the solver settings
  (`ZANDPACK_HBAR`, `MPLBACKEND`, `RK_method`).
* **pytest.** A folder named `tests/` is what `pytest` looks for, but the heavy
  suite must not be collected by accident (a full run is ~34 min and up to
  ~11 GB per heavy test). Keep `run_zandpack_tests` authoritative, or gate it
  behind a `slow` marker.

The move is mechanical: copy `TestRUN/` to `<repo>/tests/`, add `conftest.py` +
`README.md`, adjust the `pp_folder` default, extend `.gitignore`, delete the
other three copies (leaving a README pointer if their history matters). It is
deliberately **not done here** — this working directory is the only one I may
write to.
