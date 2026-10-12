# Migration plan: fitting routine + sweep framework + grid refinement → Zandpack

Target: `Zandpack/AI_code/FittingRoutine/` — a nested PACKAGE inside
the installed Zandpack (user decision, 2026-10-10). This supersedes
the flat `AI_code/AI_fit_routine.py` layout of the earlier plans
(package-era original preserved at
`backups/migration_plan_zandpack_2026-10-08_package_era.md`).

Living INSIDE the package kills the entire import-path bug class of
this repo (the root `Zandpack/`/`Block_matrices/`/`siesta_python/`
inspection copies can never shadow the installed packages again),
and the glue may import Zandpack/siesta_python types directly.

## 1. Target layout

```
Zandpack/AI_code/FittingRoutine/
  __init__.py               empty EXCEPT the alias registration
                            (see section 2) — makes the folder a
                            package and keeps old names picklable
  AI_fit_base.py            the whole routine (~4,050 lines; was
                            AI_fit.py, renamed 2026-10-10): data
                            prep, pole init, variable projection +
                            the three solvers (lsmr default / exact /
                            gram), PSD constraints, adaptive loop,
                            fit_fixed_poles (pole inheritance),
                            validation, sampling-refinement
                            proposals, and the TD glue
                            (refine_loop / resample / extract_dense /
                            inject_fits / sampled_transmission_dos /
                            fit_boosted / equalize_pole_counts) —
                            the GRID REFINEMENT lives here
  AI_fit_sweep.py           the sweep framework (~370 lines; was
                            sweep_framework.py): GammaSweep
                            (health-gated multi-start sweeps,
                            per-(lead,ik) slices, compressed npz
                            caches), evaluate_pair (T/DOS + plots),
                            pole_inherit_fit
  AI_fit_adaptivesampling.py  MODULE-ALIAS of AI_fit_base — the
                            sampling/glue API surface (was
                            refine_sampling.py)
  AI_fit.py                 compat alias shim (relative import)
  refine_sampling.py        compat alias shim (relative import)
  AI_fit_routine_tests.py   the two ported suites (from tests/)
  docs/                     the .md material (section 4)
  validation/               the 92 gate scripts + 6 READMEs
                            (section 5; artifacts excluded)
```

Import surface: `from Zandpack.AI_code.FittingRoutine.AI_fit_base
import fit, fit_fixed_poles`, `... .AI_fit_sweep import GammaSweep`,
or the sampling surface via `... .AI_fit_adaptivesampling`.

## 2. The alias mechanics (critical for old pickles) — ONE canonical import path

CANONICAL PATH RULE: the package import
(`import Zandpack.AI_code.FittingRoutine`) is THE way in. The
FittingRoutine folder must NEVER be placed on sys.path/PYTHONPATH —
a flat `import AI_fit_base` from a sys.path'd folder would create a
SECOND module object (same classes under a different __module__
name), and pickles written under one identity would silently fail
under the other. There is exactly one module object; every name is
a sys.modules alias of it:

```python
# Zandpack/AI_code/FittingRoutine/__init__.py
import sys as _sys
from . import AI_fit_base as _base
# historical + flat-name aliases, ALL pointing at the ONE package
# module object (setdefault: never overwrite an existing entry):
_sys.modules.setdefault("AI_fit", _base)           # pickle compat
_sys.modules.setdefault("AI_fit_base", _base)      # flat-name compat
_sys.modules.setdefault("refine_sampling", _base)  # historical name
_sys.modules.setdefault("AI_fit_adaptivesampling", _base)
```

After `import Zandpack.AI_code.FittingRoutine` (once, at the top of
any driver), every old `import AI_fit` / `import refine_sampling`
and every post-merge-cache unpickle resolves to the one object. The
two alias FILES (AI_fit.py, refine_sampling.py) use RELATIVE
imports (`from . import AI_fit_base as _base`) and exist only for
in-folder convenience; the repo-root versions of the aliasing
pattern are the verified template (alias identity, cache
unpickling, both suites ALL PASS, 2026-10-10).

PICKLE PORTABILITY, SCOPED: post-merge caches (module names
'AI_fit'/'refine_sampling') are covered by this alias layer.
PACKAGE-ERA caches (module 'lorentz_fit.FitResult') are NOT
portable and NOT covered: the legacy lorentz_fit/ package
deliberately stays in this repo (see section 6), so in the
Zandpack tree those caches fail to unpickle BY DESIGN and must be
regenerated. Docstrings state this; the compatibility guarantee is
post-merge caches only.

## 3. Dependencies

Machine-readable pin: `requirements.txt` travels with the package
(scipy==1.18.1 — the gram solver uses the private
`scipy.optimize._lsq.common`, verified on 1.18.1; numpy>=2.0;
numba and matplotlib commented as optional). The pin is ENFORCED
in code, not just documented: AI_fit_base warns loudly at import
when OMP_NUM_THREADS != '1' (threaded BLAS is slower AND breaks
reproducibility), warns loudly when gram runs on an unverified
scipy even if the private import succeeds, and raises a clear
RuntimeError when the private API is missing (tr_solver='gram'
then unavailable — use lsmr/exact). numba is optional
(bitwise-identical numpy fallbacks for the Jacobian + PSD/ADMM
kernels). sisl appears only in per-system data scripts, never in
the library.

## 4. Documentation migration (small .md files only; no large data)

MIGRATE to `FittingRoutine/docs/` (total ~90 KB):

| file | size | role |
|---|---|---|
| GUIDE_3py_integration.md | 27 KB | the complete user guide: 3.py template, refinement recipes, solver table, pole inheritance (8), sweep usage (9), pitfalls |
| never_form_J_guide.md | 16 KB | the performance guide that drove the gram solver |
| varproj_factors_speedup_guide.md | 14 KB | the variable-projection factorization guide |
| speedupguide_new.md | 8 KB | the channel-compression/basis-apply guide |
| lsmr_vs_exact_edge_band.md | 8 KB | the lsmr edge-band bias measurement |
| performance_guides_outcome.md | 6 KB | what worked / what did not |
| implementation-plan-...md | 9 KB | the original design document (history) |

PLUS into the AI_fit_base MODULE DOCSTRING (not a file): the
distilled HANDOFF content — the 4.16 high-quality-fit recipe, the
4.15 framework reference, the 6.4 knob table, the pitfalls
(n_res=0 trap, metric mixing, matched eta, health gate).

DO NOT MIGRATE: `HANDOFF.md` (135 KB — the PROJECT record of THIS
repo: layout, history narratives, session log; the distilled content
travels via the docstring instead), `AGENTS.md` (this repo's agent
instructions), `project_handoff.md` (retired stub), the per-system
`Data2/<sys>/README*.md` round records (they document THIS repo's
runs and reference local caches; keep a copy as reference if
wanted).

## 5. Validation: two tiers — CI-able tests vs data-dependent provenance

`validation/` here is 20 MB but only ~480 KB is migratable
material: 92 gate/benchmark .py scripts + 6 READMEs. Migrate those
into `FittingRoutine/validation/` preserving the 15 numbered
folders. EXCLUDE (regenerable, and mostly pickle-referencing old
module names): all `*.npz`, `*.jsonl`, `*.prof`, `*.png`, pickles
— 19 MB in 11_gram_basin_spread alone.

BE HONEST ABOUT WHAT THEY ARE: the gates split into two tiers.

- CI-ABLE (runnable from a fresh clone, synthetic data only):
  `AI_fit_routine_tests.py` (the two ported suites — recovery,
  Jacobian, PSD, off-window, model form), and the self-contained
  unit gates: `10_gram_solver` (identities/step agreement on
  synthetic cases), `13_refit_scales/unit_gate.py`,
  `14_psd_numba_fusion/unit_gate.py` (bitwise kernel gates). These
  are the post-migration CI story and they are strong: optimizer,
  Jacobian, gram-solver, PSD/ADMM and NNLS gates are all synthetic.
- DATA-DEPENDENT PROVENANCE (NOT runnable without the Data2
  pickles, which do not migrate — section 6): every gate that
  reads `Data2/<sys>/tmp.gz` or a README round record. These
  scripts document HOW the nine-system numbers were produced;
  they cannot be wired into CI and the numbers are not
  regenerable from the Zandpack tree alone. Their READMEs must
  say so; cite the numbers as provenance, not as a repo-runnable
  gate.

Run pattern after the migration: gates import the PACKAGE (the
canonical path, section 2) — never put the FittingRoutine folder
itself on sys.path. Adapt each gate's import to
`import Zandpack.AI_code.FittingRoutine as fr; lf = fr.AI_fit_base`
(or the AI_FIT_MODULE env-var pattern the tests use).

## 6. What does NOT migrate (stays in this repo)

`Data2/` (GB of pickles + per-system sweep caches — workflow
examples, not library code; the per-system pipeline scripts are
ready-made TEMPLATES to copy when applying the routine to a new
system), `bench/` (package-era benchmarks, data-heavy), `attempt-1/`,
`checkpoint_*/`, `backups/` (this repo's snapshot history),
`lorentz_fit/` (legacy; stays HERE so package-era caches keep
unpickling in THIS repo — CONSEQUENCE: package-era caches are NOT
portable to the Zandpack tree and must be regenerated there; the
section-2 alias guarantee is post-merge caches only), `.junk/`,
`Archive.tar.gz`, `AGENTS.md`, `HANDOFF.md`, `requirements.txt`'s
comment header stays authoritative for the pin.

## 7. Key facts the migration must respect (all measured, HANDOFF 4.11-4.16)

1. scipy pin (section 3) or keep gram opt-in-with-guard.
2. Pickle module-name trap (section 2): register the old names in
   __init__.py BEFORE any cache unpickle.
3. fit_boosted(n_res=0) runs an UNCAPPED residual fit — "no boost"
   is fit() or GammaSweep(n_res=None), never n_res=0.
4. Metric warning: fit_boosted returns CombinedFit (.rms =
   relative FROBENIUS), fit/fit_fixed_poles return FitResult (.rms
   = quadrature on scaled data) — never compare across.
5. Matched eta: the comparison/evaluation eta MUST equal the
   contour eta (three stored suite comparisons were inflated ~5x by
   the mismatch, which also changed their rankings).
6. Health gate + multi-start: single-start results on
   basin-sensitive systems carry no information; the production
   recipe is 3-6 jittered starts judged by padded min eigenvalue AND
   sample rms together; calibrate rms_max per system.
7. Init mode is a no-op under psd=True (the dispatch always uses the
   NNLS PSD dictionary); inert knobs: local_floor_frac, ridge,
   prune_rel_tol; strongest cross-system default: mult=1 +
   w_max=0.3, tight target 1e-4.
8. Ag3Au3-family 'exact' pin RESOLVED: gram + the multi-start health
   gate beats (Ag3Au3) or ties (gap) the pinned exact at ~5x less
   wall — migrate the pin OUT.
9. Sweep caches are compressed npz; the framework resumes per
   (config, start).
10. Grid refinement: the recommended mode is refine_loop(
    derivative_only=True, do_remove=False) — the removal round is
    counterproductive on deliberately dense sharp-feature grids;
    RSSE-style systems cannot use the in-place loop (rebuild-folder
    pattern).

## 8. Steps (each ends with a green gate)

1. Create `Zandpack/AI_code/FittingRoutine/` + `__init__.py` with
   the alias registration; copy AI_fit_base.py, AI_fit_sweep.py,
   AI_fit_adaptivesampling.py; add the two relative-import alias
   shims.
2. Port the tests as AI_fit_routine_tests.py (they take the import
   target from AI_FIT_MODULE):
   `AI_FIT_MODULE=Zandpack.AI_code.FittingRoutine.AI_fit_base`
   Acceptance: ALL PASS, recovery numbers identical to this repo.
3. Pickle-compat gate: load one old cache from this repo with the
   package imported (old names registered); evaluate it.
4. Copy docs/ (section 4) + validation code (section 5); move the
   distilled HANDOFF content into the module docstring.
5. Numerical identity gates: tr_solver='exact' pinned ->
   bit-for-bit vs the stored checkpoints; lsmr default -> the
   relaxed gate (same-or-better Gamma rms, PSD intact, T(0)/bands
   within tolerance); production recipe -> gram@1e-9 + multi-start
   health gate, expecting BETTER matched-eta T rms than the stored
   numbers on all nine tested systems. Cheapest end-to-end checks:
   agnr and VerySmallRSSE.
6. Port the Data2 driver pattern to the package import; keep the
   old R.Fit/curvefit_all/iterative_PSD path (baseline + other
   workflows use it) — the new routine is additive.
7. Keep this repo as the reference implementation until Zandpack's
   copy has reproduced a full system's numbers at the matched eta.

## 9. Risks and mitigations

| risk | mitigation |
|---|---|
| dual module identity (flat import of AI_fit_base creates a second class set) | ONE canonical path (section 2): never put the folder on sys.path; __init__ registers ALL flat names as aliases of the one package module |
| gram's private scipy API breaks on upgrade | requirements.txt pin; loud import-time warning on unverified scipy; hard RuntimeError when the API is missing |
| threaded BLAS nondeterminism | import-time RuntimeWarning when OMP_NUM_THREADS != 1 + docstring rule |
| old pickles stop unpickling in the package | __init__.py alias registration (section 2) + gate in step 3; package-era caches are explicitly NOT covered (regenerate) |
| basin chaos makes a "migrated" fit look wrong | never single-start; multi-start health gate is part of the framework; >30 min fit = pathological basin, stop and reseed |
| numba absent on the target | automatic numpy fallbacks, bitwise identical |
| eta-mismatched comparisons mislead the acceptance gate | evaluate at the contour eta (step 5) |
| validation gates assume repo-root layout | canonical-path import adaptation (section 5); CI = the synthetic tier only |
| Zandpack evolves (containers, setLval) | the glue is one ~350-line section, isolated |
