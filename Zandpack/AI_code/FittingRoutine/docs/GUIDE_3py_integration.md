# Guide: running the new fitting routine from your 3.py scripts

RENAME (2026-10-10): the routine module is `AI_fit_base.py` (was
AI_fit.py), the sweep framework is `AI_fit_sweep.py` (was
sweep_framework.py), the sampling/glue surface is
`AI_fit_adaptivesampling.py` (was refine_sampling.py). The old
names survive as module-alias shims. This guide uses the NEW names.

PHYSICS LAYER (2026-10-13/14): the routine now PREVENTS the
monster-residue pathology at the source (an in-optimizer Tikhonov
ridge, `opt_ridge`, DEFAULT 1e-3) and backstops it with a guarded
residue cap (`residue_cap_mult=10`, on by default). Section 4b is
the entry point; the evidence lives in HANDOFF 4.16 items 11-12
and `Data3/default_runs/`.

Audience: you, porting a system folder in `Data2/<system>/` (or a fresh
one) from the old `PoleGuess -> R.Fit -> curvefit_all -> iterative_PSD`
pipeline to the new `AI_fit_base.py` routine — by editing that folder's
`3.py`. This guide is self-contained: everything you need is here or
in `HANDOFF.md` 6.4 (tuned knobs) and 4.9-4.14 (why the knobs are
what they are).

What you write is ONE script with this shape:

```
load pickles -> (optional) refine sampling -> fit -> inject -> old tail
```

Everything after `inject` (find_correction, Renormalise_H, tofile,
Inspect_transmission_from_hilbert_transform) is your existing 3.py
tail, byte for byte.

## 1. What replaces what

| old script block | new call | notes |
|---|---|---|
| 2.py's fixed contour | `lf.refine_loop(R, D, ...)` | OPTIONAL (section 6): adaptive energy grid — adds points where the sampling under-resolves, drops redundant ones |
| `R.PoleGuess(...)` per lead | inside the fit | deterministic `skrf` PSD-dictionary init (`n_init` poles) |
| `pad_poles` / `fix_missing` / `nlmax` bookkeeping | inside the fit | adaptive pole insert/remove/merge; pole cap = `max_poles` |
| `R.Fit(...)` + `R.curvefit_all(...)` | `lf.fit_boosted(E, F, kw, n_res=8)` | shared-pole Lorentzian fit + residual-correction fit + combined-model residue-cap check |
| `iterative_PSD(...)` per lead | inside the fit (`psd=True`) | exact PSD projection + PSD-constrained ADMM solve |
| `R.NO_fitted_lorentzians[...]` containers | `lf.inject_fits(R, fits)` | also pads pole counts to a common `NumL` and runs `FitNO2O` |
| your tail | unchanged | — |

The injected model is the SAME Lorentzian form the Zandpack machinery
already evaluates (`w^2 / ((E-e)^2 + w^2)`, per-ik pole arrays), so
nothing downstream changes.

## 2. Standing environment rules (break these and you lose hours)

1. `OMP_NUM_THREADS=1` — measured faster AND required for
   run-to-run reproducibility (threaded BLAS shifts the fit basin).
   The ambient shell already exports it; if you launch from elsewhere,
   set it FIRST, before importing numpy.
2. Load pickles BEFORE any `sys.path` insert that touches this repo —
   the repo-root `Zandpack/`, `Block_matrices/`, `siesta_python/`
   folders are inspection copies that SHADOW the installed packages.
   (This is why the import block below inserts the root, imports, and
   removes it again — keep that order.)
3. Keep `FittingRoutine/AI_fit_base.py` as the single source; do not copy
   pieces of it into your 3.py.

## 3. The template (copy, then adjust only the knobs block)

```python
# --- new 3.py --------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy

import pickle, gzip, sys
import numpy as np

# (0) loads FIRST (shadow rule, section 2)
R = pickle.load(gzip.open('tmp.gz', 'rb'))
D = pickle.load(open('Device.SiP', 'rb')); D.mpi = "mpirun "
orbs = [...]          # exactly as your 2.py selects orbitals

# (1) import the routine (temporary root insert, removed at once)
ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, ROOT); import AI_fit_base as lf; sys.path.remove(ROOT)

# (2) knobs -- GENERIC STARTING POINTS; the per-system tuned table is
#     HANDOFF 6.4. Measure before trusting these on a new system.
BASE_KW = dict(
    n_init=24, max_poles=100, target_rms=2e-3,
    psd=True,
    psd_force=True,        # False if the sampled Gamma has a deep
                           # narrow notch (PSD-forcing fills notches;
                           # HANDOFF 4.9, VerySmallRSSE)
    eta_raw=1e-2,          # = your contour's imaginary part
    eta_floor_mult=5.0,    # SYSTEM-DEPENDENT: 5 for zgnr + metals,
                           # 1 for agnr + the eta=0.02 systems
                           # (per-system table: HANDOFF 6.4)
    w_max_raw=0.5,         # width ceiling; 0.35 only if slow T
                           # undulation persists (4.9)
    local_floor_frac=1.5,
    tr_solver="lsmr",      # default; see solver note below
    # --- physics layer (section 4b); both DEFAULT-ON, listed here
    #     so you know they exist ---
    opt_ridge=1e-3,        # Tikhonov ridge inside every fit_poles
                           # stage + the final solve. Prevents the
                           # near-degenerate-pole residue blow-ups
                           # (Ag3Au3: 0/6 -> 6/6 healthy starts).
                           # Known-CLEAN systems (zgnr_spin, nogap,
                           # zgnr) take opt_ridge=None for max
                           # accuracy (they pay T for the cure).
    residue_cap_mult=10.0, # backstop: ||C_i||_F <= 10 x max||F_n||;
                           # guarded, never silently enforced - a
                           # failure leaves the fit unchanged with
                           # history['residue_cap_failed'] + a loud
                           # warning (read it: the fit's accuracy is
                           # then riding on cancelling residues)
)

# basin-sensitive system? (Ag3Au3 measured; any system whose single
# start gives bad padded min-eig or oddly large rms) -> set True.
# Costs 3 fits per (lead, ik) instead of 1; with tr_solver="gram"
# that is still ~6x cheaper than one 'exact' fit.
MULTI_START = False

# (2b) OPTIONAL: refine the sampling grid FIRST (full recipe: section 6).
# Set True when the fit rms is limited by the GRID, not the model —
# the loop reruns TBtrans (n_rounds + 1 times), settles the grid, and
# leaves the refined sampling inside R (pickle it right after).
RUN_REFINE = False
if RUN_REFINE:
    R.Device = D                       # refine_loop drives TBtrans via D
    out = lf.refine_loop(
        R, D,
        read_kwargs=dict(sub_orbital=orbs, less_memory=False),
        n_rounds=3, insert_tol=5e-3, max_add_per_round=48,
        do_remove=True, remove_tol=1e-2,
        fit_kw=dict(BASE_KW),           # the loop's own per-round/final fits
        eta=1e-2,                      # = your contour's imaginary part
        # derivative_only=True,         # fit-free insertion rounds (cheap
        #                                 pre-pass; section 6.4)
        # runner=lambda R_, D_: D_.mimic_tbtrans_spin(R_.Contour),
        # eta=2e-2,                    # spin systems: section 6.3
    )
    with gzip.open('tmp_refined.gz', 'wb') as f:
        pickle.dump(R, f, protocol=4)   # KEEP the refined R
    # Option A: inject the loop's own final fits (single-start plain
    # fits, fit_kw as passed) and skip stage (3):
    #   fits = out['fits']
    # Option B (recommended): fall through to stage (3) — your own
    #   production fit (fit_boosted / multi-start health gate) on the
    #   refined grid.

# --- multi-start health-gate helpers (path B only) ------------------
def _init_poles(E, F, n_init=24, e_margin=0.15):
    """The deterministic init poles the unseeded fit would use."""
    fd = lf.prepare(E, F)
    c0, w0 = lf._poles_init_dispatch("skrf", fd.E, fd.F, n_init,
                                     e_margin=e_margin, psd=True)
    return c0, w0, fd

def jittered_starts(E, F, n_init=24, n_pert=2, jitter=1e-6, seed=1):
    """Control start + n_pert copies with every init pole center AND
    width jittered by U(-jitter, jitter) eV (raw -> scaled units)."""
    c0, w0, fd = _init_poles(E, F, n_init)
    rng = np.random.default_rng(seed)
    starts = [("c0", np.asarray(c0, float), np.asarray(w0, float))]
    for s in range(1, n_pert + 1):
        de = rng.uniform(-jitter, jitter, c0.size) / fd.e_scale
        dw = rng.uniform(-jitter, jitter, c0.size) / fd.e_scale
        starts.append((f"s{s}", np.asarray(c0, float) + de,
                                np.asarray(w0, float) + dw))
    return starts

def fit_health(r, E, pad=2.0, npts=801):
    """Health record: RELATIVE sample rms AND padded min eigenvalue.
    Judge a fit by BOTH, never the sample rms alone."""
    Eg = np.linspace(min(E) - pad, max(E) + pad, npts)
    return dict(rms=float(r.rms),
                min_eig=float(np.linalg.eigvalsh(r.evaluate(Eg)).min()))

def is_healthy(rec, min_eig_tol=-1.0, rms_max=0.05):
    return rec["min_eig"] > min_eig_tol and rec["rms"] <= rms_max
# Thresholds: from the Ag3Au3 gram production run (min_eig -0.15,
# rms 0.005-0.014 per fit). min_eig_tol=-1 is a physical floor (below
# that the model leaves the PSD cone between samples); rms_max should
# sit a few x above whatever target_rms the system actually supports
# (HANDOFF 6.3) so it rejects catastrophic basins, not honest fits.
# ---------------------------------------------------------------------

# (3) the fit
fits = {}
n_ik = len(R.Nonortho_Gammas[0].vals[0])
for lead in range(R.num_leads):
    for ik in range(n_ik):
        E, F = lf.extract_dense(R, lead, ik)
        kw = dict(BASE_KW,
                  channel_tol=1e-4 if F.shape[1] > 32 else 0.0)
        if MULTI_START:
            cands = []
            for tag, c0, w0 in jittered_starts(E, F):
                r = lf.fit_boosted(E, F, dict(kw, init_poles=(c0, w0)),
                                   n_res=8)
                rec = fit_health(r, E); rec.update(tag=tag, r=r)
                cands.append(rec)
                print(f"lead{lead} ik{ik} {tag}: rms {rec['rms']:.4g} "
                      f"n={r.n_poles} minEig {rec['min_eig']:+.3g}")
            good = [c for c in cands if is_healthy(c)]
            best = min(good or cands, key=lambda c: c["rms"])
            if not good:
                print(f"WARNING lead{lead} ik{ik}: no healthy start, "
                      f"using best-of-bad {best['tag']}")
            fits[(lead, ik)] = best["r"]
        else:
            r = lf.fit_boosted(E, F, kw, n_res=8)
            rec = fit_health(r, E)
            print(f"lead{lead} ik{ik}: rms {rec['rms']:.4g} "
                  f"n={r.n_poles} minEig {rec['min_eig']:+.3g}")
            fits[(lead, ik)] = r

# (4) inject into the Zandpack containers (pads pole counts to one
#     common NumL, runs FitNO2O)
lf.inject_fits(R, fits)

# (5) your old 3.py tail, unchanged:
from Zandpack.FittingTools import find_correction
C = find_correction(R, Emin=-3, Emax=3.0)
R.Renormalise_H(C)
R.tofile('src', tol=1e-3, write_ixi=False)
R.Inspect_transmission_from_hilbert_transform(np.linspace(-11, 11, 550),
                                              eta=1e-2, figure="Tfit.svg")
```

Notes on the template:

- `R` is MUTATED by `inject_fits`. If you want to re-run or compare,
  re-load `tmp.gz` into a second object first (see the verification
  snippet in section 5) or re-pickle before injecting.
- `n_ik` for single-k systems is 1; the loop is harmless either way.
  Multi-k (spin) systems need nothing extra — `inject_fits` guarantees
  the common pole count Zandpack's `R.NumL` helpers require.
- `jittered_starts` mirrors `validation/11_gram_basin_spread/
  spread_common.py` (make_starts + baseline_init_poles). The 1e-6 eV
  jitter is deliberately tiny: it does not change the physics, it only
  samples which ATTRACTOR BASIN the adaptive loop falls into (measured
  on Ag3Au3: identical-start runs differ as much as different solvers
  do; HANDOFF 4.13).

## 4. Solver choice (`tr_solver`)

| value | when | measured character (HANDOFF 4.11/4.14) |
|---|---|---|
| `"lsmr"` (default) | general use | 1.5-1.8x faster than exact, smoother residuals, better notch fidelity; same-sign bias at steep band edges that accumulates in integrated quantities |
| `"exact"` | integrated currents across band edges; bit-for-bit reproduction of pre-2026-10-09 stored numbers | slowest; also basin-sensitive in the adaptive loop |
| `"gram"` | fastest; the production recipe for basin-sensitive systems together with MULTI_START | 2-8x faster than exact, tracks exact to ~2.5e-14 per step; needs scipy PINNED at 1.18.1 (private-API guarded import) |

`gram` is opt-in and pairs with `gram_cut=1e-9` (the default; measured
bit-identical to cut=0, while 1e-7 is actively harmful on Ag3Au3). A
basin-sensitive recipe:

```python
BASE_KW = dict(..., tr_solver="gram", gram_cut=1e-9)
MULTI_START = True     # 3 starts/fit + health gate (section 3)
```

That is the recipe that produced Ag3Au3 T rms 0.0077 (old 3.py
pipeline: 0.0383) in ~3.5 min for 6 fits, where a single 'exact'
control start once landed T rms 3.44.

NOTE (2026-10-14): under the ridge, all three solvers AGREE on
healthy fits (Ag3Au3: gram 0.00446 / lsmr 0.00441 / exact 0.00447)
- the historical solver gaps were monster-basin artifacts. Choose
by speed (gram fastest at large M*n_ch), not by fear.

## 4b. The physics layer: `opt_ridge` and the residue cap

THE PROBLEM: near-degenerate pole clusters give the pole search
near-null LS directions where coefficients explode - residues up to
1e11x the sampled Gamma scale, Zandpack's per-pole eigenvalue print
shows +-1e8, and the transport tail inherits garbage. Historically
only the final polish carried a ridge, so the POLE SEARCH was free
to reward cancellation (excellent sample rms, unphysical model).

THE FIX (in order of effect, both DEFAULT-ON):

1. `opt_ridge=1e-3` — a Tikhonov ridge inside EVERY fit_poles
   stage and the final coefficient solve (the varproj derivative
   stays exact). The clusters never form: Ag3Au3 went 0/6 -> 6/6
   healthy starts at BETTER frob and T 0.0062 (monster era) ->
   0.0044; the Fig2Ag5 tight-target pileup (48 added poles,
   1e8-residue cluster) never forms (32 added) at better frob and
   better 2-4 eV band; per-pole eigenvalues return to O(data
   scale).
2. `residue_cap_mult=10` — the backstop: ||C_i||_F <= 10 x
   max_n||F_n||_F, enforced after the PSD stage by a guarded
   drop/clip re-solve with a BAND-level guard (a global rms
   threshold is blind to damage concentrated in one energy band).
   Never silent: if no bounded state passes the guard, the input
   fit is returned UNCHANGED with `history['residue_cap_failed']`
   plus a loud warning. SEEING THAT WARNING MEANS: this fit's
   accuracy is riding on cancelling residues - either accept the
   flag, or change something upstream (the ridge should have
   prevented it; if you turned it off, put it back).

PER-SYSTEM LAMBDA RULE (measured, Data3/default_runs): the ridge
is a TREATMENT, not a vitamin. Keep the default 1e-3 unless the
system is known-clean; for maximum accuracy on known-clean systems
(nogap, zgnr_spin, zgnr - census residue ratios ~1) use
`opt_ridge=None` (measured cost of the default there: zgnr_spin T
0.139 -> 0.212, nogap 0.068 -> 0.106). Pathological families
(Ag3Au3*, tight-target configs) NEED the ridge - do not disable it
there.

WHAT TO READ IN `fit.history` (recorded on every fit, free):
- `max_pole_amplification`, `design_kappa` — VULNERABILITY
  metrics (how ill-conditioned the design is). Healthy systems
  still show large values (1e3-1e7) — do not gate on them.
- `cancellation_index` — the EXCITATION metric (sum of in-window
  per-pole contributions / max Gamma norm). Healthy ~1-3;
  monsters 1e5+. Gate on THIS if you gate on anything.
- `ridge_used` — the FINAL solve's effective ridge (including
  its >2x fallback to 0). Sanity: equals opt_ridge when set.
- `residue_cap_failed`, `n_dropped`, `n_clipped`,
  `target_missed_due_to_cap` — the cap verdicts.
- `psd_applied` — whether the PSD model was actually kept.

The sweep framework's `fit_health`/`is_healthy`
(AI_fit_sweep.py) now also carries and gates the padded-grid
ENVELOPE (max||Gamma||/max||F|| on a grid padded beyond the
window, default tolerance 5) — use it instead of hand-rolling
health checks in new drivers.

## 5. Verify (do not skip on a new system)

Per fit, read the health record printed in the template:
`rms` should be at/below `target_rms` (or the best the system
supports, HANDOFF 6.3), `min_eig` should be a small negative number
or positive; min_eig << -1 means the model leaves the PSD cone
between samples and the transport tail can go unphysical. ALSO
read the physics-layer history keys (section 4b): a healthy
production fit has `residue_cap_failed` absent/False and
`cancellation_index` in the 1-3 range; `residue_cap_failed=True`
means the residues stayed unbounded and the accuracy is riding on
cancellation. And check what Zandpack prints per pole: the
eigenvalues of the Lorentzian Gammas should be O(data scale), not
+-1e8.

System level, compare the fitted T against the sampled self-energy
reference BEFORE injection:

```python
R = pickle.load(gzip.open('tmp.gz', 'rb'))      # pristine copy
E_ref, T_ref, _ = lf.sampled_transmission_dos(R)
# ... fit + inject into this same R (or a re-loaded copy), then:
Eg  = np.linspace(-5, 9, 600)                   # your window
_, T_new = R.Inspect_transmission_from_hilbert_transform(
    Eg, eta=1e-2, return_result=True)
t_rms = float(np.sqrt(np.mean(
    (T_new.real - np.interp(Eg, E_ref, T_ref)) ** 2)))
```

Reference values to beat/expect: per-system table in HANDOFF 6.3
and the POST-RIDGE baselines in HANDOFF 4.16.12
(Data3/default_runs/summary.json): Ag3Au3 0.0044, Ag3Au3_gap
0.0034, spinpol 0.0081, Fig2Ag5 0.0061 [tgt1e4], agnr 0.0221
[tgt1e4], zgnr_spin 0.139 [noridge], nogap 0.068 [noridge], zgnr
0.115 [noridge]. NOTE: the older Ag3Au3-family numbers (0.0077/
0.0062/0.0036 era) were produced by MONSTER fits (residues 1e5x
data); do not chase them at the cost of the physics layer.

Full end-to-end reference implementations you can diff against:
`Data2/zgnr/compare_dos_t.py` (old and new pipelines side by side),
`Data2/zgnr_spin/compare_dos_t.py`,
`Data2/VerySmallRSSE/refined/compare_dos_t.py`,
`validation/11_gram_basin_spread/ag3au3_gram_production.py`
(the MULTI_START + gram recipe as actually run).
Sampling-refinement drivers: `Data2/zgnr/refine.py` (standard),
`Data2/zgnr_spin/refine.py` (spin), `Data2/VerySmallRSSE/refined/
refine.py` (RSSE rebuild pattern).
Generic driver for any system: `Data2/apply_method.py`.

## 6. Full sampling grid refinement (`refine_loop`)

Do this BEFORE the production fit (template stage 2b). When to bother:
the fit rms is stuck well above `target_rms` while the pole count is
nowhere near `max_poles` — that is the grid under-resolving the
function, not the model lacking poles. When in doubt, compare a fit's
rms on the stored grid against the same fit on a x2-denser hand-made
grid; if the dense grid fits much better, refine.

### 6.1 What the loop does

Per insertion round (default 3): fit every (lead, ik) with `fit_kw`,
propose insertions where the piecewise-linear and 4-point-Lagrange
interpolants disagree (plus fit-residual peaks and barely-resolved
poles), take the UNION over leads (TBtrans evaluates all leads at
every contour point anyway), rebuild the contour, rewrite the fdf,
rerun TBtrans, read. After the insertion rounds, one removal round
drops points the leave-one-out test flags as redundant for EVERY
(lead, ik). Then the final fits run once, on the settled grid.

Returns `{fits, history, removed, lines}`:
- `fits` — {(lead, ik): FitResult} at the final grid (plain single-start
  `lf.fit` with `fit_kw`, NOT `fit_boosted` and no multi-start gate);
- `history` — per-round record (grid sizes, per-(lead,ik) rms);
- `removed` — indices dropped in the removal round;
- `lines` — the final per-lead energy arrays.

`R` comes out MUTATED: the refined sampling is now the stored sampling
(and `tmp.gz` on disk is untouched — that is your pre-refinement
reference). Pickle the refined object under a NEW name
(`tmp_refined.gz`) immediately; that pickle is what later 3.py runs
load.

### 6.2 The knobs

| kwarg | default | meaning / when to change |
|---|---|---|
| `read_kwargs` | required | must match your 2.py's `R.read_data` call — the sub-orbital selection above all |
| `initial_lines` | None | None = refine the stored sampling as-is. A coarsened copy (e.g. `line0[::4]`) starts cheap and lets the loop place density where needed (zgnr: 63-pt start -> 100 refined vs 251 original) |
| `n_rounds` | 3 | insertion rounds; the loop stops early if a round proposes nothing |
| `insert_tol` | 5e-3 | relative midpoint-interpolant discrepancy that flags an interval for splitting |
| `max_add_per_round` | 48 | cap, split fairly over (lead, ik) |
| `do_remove` / `remove_tol` | True / 1e-2 | the leave-one-out redundancy test (edges are never dropped) |
| `fit_kw` | workflow defaults | the loop's own fits; `eta_raw` is auto-set from `eta`. Plain `fit` defaults already carry psd_force=True, w_max_raw=0.5, eta_floor_mult=5.0 — pass `dict(BASE_KW)` to stay consistent with your production knobs |
| `eta` | 1e-2 | MUST equal your contour's imaginary part (spin systems: 2e-2) |
| `derivative_only` | False | fit-free insertion mode (6.4) |
| `runner` | None | replaces the TBtrans evaluation step (spin systems, 6.3) |

### 6.3 System-type recipes

**Standard TBtrans systems** (zgnr pattern — the template's stage 2b
verbatim). Reference driver: `Data2/zgnr/refine.py`.
Required: `R.Device = D` before the call, `read_kwargs` from 2.py.

**Spin-polarized systems** (zgnr_spin pattern — no TBtrans binary; the
self-energies come from `D.mimic_tbtrans_spin`, two k-points per lead).
Run the folder's `sample.py` once first if you have not; then:

```python
line0 = np.asarray(R.Contour[R.sampling_idx[0]]).real
coarse = line0[::2].copy()      # coarsened start grid
out = lf.refine_loop(R, D,
    read_kwargs=dict(sub_orbital=orbs, less_memory=True),
    n_rounds=3, initial_lines=[coarse], insert_tol=5e-3,
    max_add_per_round=64, do_remove=True, remove_tol=1e-2,
    eta=2e-2,
    runner=lambda R_, D_: D_.mimic_tbtrans_spin(R_.Contour))
```

Reference driver: `Data2/zgnr_spin/refine.py`.

**RSSE systems** (VerySmallRSSE pattern — real-space electrode
self-energy): the in-place `resample` does NOT apply; every round
rebuilds the whole device by re-running a parameterized
`make_device.py` with an updated `energies.npy` and reloading the
pickle. That loop is hand-rolled per system — copy
`Data2/VerySmallRSSE/refined/refine.py` (its `rebuild(energies)` /
`fit_all` / `propose` pattern) and adapt. Do not attempt
`refine_loop` here.

### 6.4 Cheap variant: derivative-only pre-pass

`derivative_only=True` (with `deriv_tol=3.0`) runs NO fits during the
insertion rounds: proposals come purely from
d(sum_ij |F_ij|)/dE, so steep flanks (band edges, notch walls) get
density at a fraction of the cost. The removal round is fit-free in
both modes; the FINAL fits still run once with `fit_kw`. Use it
standalone when per-round fits are too expensive, or as a pre-pass
before a fit-based loop. If you run your own stage-(3) production fit
afterwards anyway, the loop's final fits are only a sanity readout —
`fit_kw` can carry reduced knobs (small `n_init`, low `max_poles`) to
cut that cost too.

CAUTION with the removal round (measured on nogap_Ag5, 2026-10-09):
on systems whose sharp-feature region is DELIBERATELY sampled densely
(e.g. a van Hove streak at 0.009 spacing), the leave-one-out
redundancy test flags exactly the points the sharp structure needs —
removal dropped 73/187 points at tol 1e-2 and 175/522 at tol 1e-3,
degrading the out-of-sample rms both times (0.371/0.103) while the
gradient-based variant WITH `do_remove=False` improved it 2-4x and
placed all 96 insertions inside the streak. For such systems use
`derivative_only=True, do_remove=False` and keep the 2.py grid
density.

### 6.5 Grid verification (do this before trusting the refined R)

- Out-of-sample check: evaluate each fit on the ORIGINAL dense grid
  (`E0, F0` snapshotted before the loop, exactly as
  `Data2/zgnr/refine.py` does) — the fit must reproduce the original
  samples between the refined points, not just on them:

```python
rms_ref = lf.frobenius_rms_error(res.evaluate(E0), F0)   # relative
```

- Watch `out['history']`: grid size per round and per-(lead, ik) rms
  should fall monotonically-ish and flatten; a grid that keeps growing
  without rms improvement means `insert_tol` is below the noise of the
  data.
- Wall: one TBtrans evaluation per round (plus one after removal) on
  top of the fits — on expensive systems prefer `derivative_only` or
  fewer rounds.

## 7. Pitfall checklist

- OMP_NUM_THREADS=1 set before numpy (section 2.1).
- Pickles loaded before the ROOT sys.path insert (section 2.2).
- `psd_force=True` except for deep-notch systems (4.9).
- `eta_raw` MUST equal your contour's imaginary part.
- Refinement (section 6): `R.Device = D` before `refine_loop`; the
  loop's `eta` MUST equal the contour's imaginary part too;
  `read_kwargs` must match 2.py; RSSE systems cannot use the in-place
  loop at all; pickle the refined R under a NEW name — `tmp.gz` is
  your pre-refinement reference. On deliberately dense sharp-feature
  grids, DISABLE the removal round (`do_remove=False`) — see 6.4.
- A sweep may mix per-lead configs: fits are independent per lead, so
  the winning pair can take lead0 from one knob config and lead1 from
  another (that mix won the nogap_Ag5 sweep: T rms 0.049 vs 0.056 for
  the best single config).
- `eta_floor_mult` from the HANDOFF 6.4 table, not the generic 5.0,
  on agnr / eta=0.02 systems.
- Do not re-pickle `R` after injection under the old name unless you
  mean it — the injected fit replaces the sampled self-energy.
- `nproc` reporting 1 is the OMP export, not the machine (18 cores) —
  irrelevant for correctness, do not "fix" it.
- Single-start results on basin-sensitive systems carry no
  information — health-gate + multi-start before concluding anything
  about a solver or a knob (HANDOFF 4.13).
- `residue_cap_failed=True` is a VERDICT, not a bug: the fit cannot
  be bounded at the current settings without blowing the guard -
  fix upstream (keep opt_ridge on) or accept the flag knowingly
  (section 4b). Do not silence it.
- Per-system lambda (4b): the opt_ridge=1e-3 default is safe
  everywhere but costs T on known-clean systems; do not "fix" a
  clean system's regression by loosening OTHER knobs before trying
  opt_ridge=None.
- Parallel sweeps (joblib/loky): workers must NOT print (warnings
  to an undrained PTY block the whole pool - suppress warnings in
  workers, file-log progress); a parent-side module-valued global
  is pickled BY REFERENCE (name it AI_fit_base_local, not
  AI_fit_base). Working templates: Data3/default_sweep.py,
  Data2/Fig2Ag5/3new_par.py.

## 8. Symmetric systems: pole inheritance (`fit_fixed_poles`)

For mirror-symmetric systems the two leads share their spectral
structure. Fit ONE electrode with the full machinery, then hand the
OTHER electrode the fitted poles and solve ONLY its residue matrices:

```python
E0, F0 = lf.extract_dense(R, 0, 0)
E1, F1 = lf.extract_dense(R, 1, 0)
f0 = lf.fit_boosted(E0, F0, kw_lead0, n_res=8)     # full machinery
f1 = lf.fit_fixed_poles(E1, F1, f0.e, f0.w,
                        psd=True, psd_force=True)  # seconds
fits = {(0, 0): f0, (1, 0): f1}
```

MEASURED (2026-10-13): Fig2Ag5 saves ~2x wall at T rms 0.0061 vs
0.0067 own-fit - inheritance is the right default there. BUT
"symmetric" is a PHYSICS statement, not a fit statement: the Ag3Au3
family is NOT symmetric (different electrodes) - forcing
inheritance there inflated T 5.4x (0.0067 own-fit vs 0.0374
inherit). Check the electrodes before inheriting; when in doubt,
run both ways once and compare at the T level. Under the residue
cap, `fit_fixed_poles` enforces clip-ONLY (never drops poles) so
the inherited pole set is exactly preserved.

`fit_fixed_poles` runs the exact final-solve + PSD machinery of the
full fit at the FIXED poles (same weights, tail anchors, PSD
accuracy guard) — no initialization, no relocation, no adaptive loop.

Measured on nogap_Ag5 (2026-10-09, finer-mesh sweep): the inherited
lead1 matched or BEAT its own full multi-start fit at the T level in
4 of 5 configs (e.g. w_max 0.25: T rms 0.0432 inherited vs 0.0749
own-fit) at ~1/50 the cost — consistent poles across the leads are
physically right for a symmetric system. Inheritance is an OPTION,
not a mandate: keep the own-fit variant in the sweep and let the
T level decide per system.

Rules learned the hard way (HANDOFF, fixed-poles re-solve finding):

- `ridge` defaults to 0.0 here (NOT the full fit's 0.05): at fixed
  poles the Tikhonov penalty biases the solve whenever the pole set
  overlaps (measured: x24 rms degradation on Ag3Au3 at ridge 0.05).
- MATCH the source fit's psd policy. `psd_force=True` only when the
  source fit's PSD model was applied (nogap: yes — works perfectly,
  inherited models PSD to -1e-15). For fits whose residues rely on
  inter-pole cancellation (Ag3Au3 family keeps them unconstrained),
  use `psd=False` — forcing the PSD cone at fixed poles there costs
  up to x23 in rms (the adaptive fit would ADD poles to compensate;
  a fixed-poles solve cannot).
- METRIC WARNING: `fit_boosted` results are `CombinedFit` objects
  whose `.rms` is the RELATIVE FROBENIUS rms; `fit_fixed_poles`
  returns a `FitResult` whose `.rms` is quadrature-weighted on
  scaled data. Never compare the two attributes directly — use
  `lf.frobenius_rms_error(model.evaluate(E), F)` on both.
- A fixed-poles re-solve at a fit's OWN poles is a consistent,
  cheap POLISH (~10% rms gain on nogap) — but it does not move the
  T level once the fit is already at ~1% sample rms (transport
  error is model-shape-limited there; HANDOFF finding 3).

## 9. Sweeping: `AI_fit_sweep.py`

The multi-start / health-gate / caching / T-evaluation machinery of
the nogap_Ag5 rounds is formalized in `AI_fit_sweep.py` at the
repo root. Typical use (after the stage-0/1 loads and the AI_fit
import):

```python
import AI_fit_sweep as sf

E, F = lf.extract_dense(R, 0, 0)
sw = sf.GammaSweep(lf, E, F, cache_dir="sweep_cache",
                   base_kw=dict(n_init=24, max_poles=90, psd=True,
                                eta_raw=2e-2, w_max_raw=0.25,
                                tr_solver="gram", gram_cut=1e-9),
                   n_res=8, n_starts=3)
results = sw.run({
    "base":  {},
    "tight": dict(target_rms=1e-4),   # the highest-leverage knob on
                                     # nogap_Ag5: T 0.043 -> 0.030
    "w02":   dict(w_max_raw=0.2),
})

# T/DOS level per config (lead1 inherited; run a second GammaSweep
# on lead1 for the own-fit alternative):
for name, r in results.items():
    pair, pick, f1 = sw.best_pair_inherit(R, name=name)
    RX = pickle.load(gzip.open("tmp_refined.gz", "rb"))   # fresh R
    row = sf.evaluate_pair(lf, RX, pair, E_ref, T_ref, dos_ref,
                           teta=2e-2, corr=(-3, 3), twin=(-5, 9),
                           bands=[(-5, -3), (2, 4)],
                           plot=f"sweep_cache/fig_{name}.png",
                           E_fit=E, F_fit=F, title=name)
```

Baked-in behaviors: fit caches resume on rerun (per config/start;
files are COMPRESSED npz — use `np.savez_compressed` for any caches
you write yourself, and note the caches pickle-reference the AI_fit
module: import `refine_sampling`/`AI_fit` BEFORE loading them, never
run the loader with cwd = the repo root); configs touching
`init`/`n_init`/`e_margin` get their OWN start set (so those knobs
are actually exercised — note `init` is inert under `psd=True`, the
dispatch always uses the NNLS PSD dictionary); the health gate is
min eig AND rms AND the padded-grid ENVELOPE together (2026-10-13:
fit_health computes it, is_healthy gates it at 5x data scale, and
the fit's cancellation_index rides along in the record for optional
gating via is_healthy(cancellation_max=...)); winners whose fit
history carries residue_cap_failed can be excluded with
is_healthy-style filtering — `sf.fit_health` already carries the
cancellation index from the fit history. `evaluate_pair` mutates R,
so pass a freshly loaded copy per evaluation. The constructor's
`n_res` sets the
residual-boost poles
(fit_boosted). `n_res=None` runs the PLAIN adaptive fit (`lf.fit`,
no boost) — use that for pole budgets. WARNING: `n_res=0` does NOT
mean "no boost": `fit_boosted` pops `max_poles` from the residual
kwargs, so `n_res=0` silently runs an UNCAPPED adaptive fit of the
residual and violates any pole budget (found in the lowpole round,
HANDOFF 4.15). Also: fits from `n_res=None` are FitResult objects
(quad rms) while boosted ones are CombinedFit (frobenius rms) —
never compare their `.rms` across groups; rank by the T level.
For MULTI-K systems run ONE
GammaSweep per (lead, ik) slice with `lead_label=f"{lead}_ik{ik}"`
(so caches never collide), ideally as parallel background
processes, then `equalize_pole_counts` before `inject_fits`.
Knobs to bother sweeping first on a new system (measured on
nogap_Ag5, spinpol_ag5, agnr, Fig2Ag5 — HANDOFF 4.15): target_rms
(1e-4 wins on 3 of 4), w_max_raw (sharp, system-specific optimum),
eta_floor_mult (never above 1 on these; 1.5 is a dead zone
everywhere), max_poles, n_init — while `local_floor_frac`, `ridge`,
`prune_rel_tol`, `n_res` and the init mode proved inert. ALWAYS
evaluate at the CONTOUR eta (two of four stored system comparisons
were inflated ~5x by an eta mismatch, which even changed their
rankings).
