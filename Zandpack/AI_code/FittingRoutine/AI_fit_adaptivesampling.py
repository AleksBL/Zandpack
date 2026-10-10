"""Adaptive-sampling API surface (2026-10-10 rename).

The implementation lives in the single file AI_fit_base.py (renamed
from AI_fit.py; merged from the lorentz_fit package + the historical
refine_sampling.py; see HANDOFF 4.10 and migration_plan_zandpack.md).
This module is an ALIAS of AI_fit_base (same module object) — the
sampling/glue surface (extract_dense, inject_fits, resample,
refine_loop, sampled_transmission_dos, fit_boosted,
fit_fixed_poles, equalize_pole_counts) is importable from here for
readability, and every other attribute of the base module is
reachable too.

The OLD import path `refine_sampling` survives via the repo-root
compat alias refine_sampling.py — needed by ~90 driver/diagnostic
scripts and by pickled fit caches, whose objects reference the
module NAME 'refine_sampling'.

Pickle portability, explicitly scoped (verified 2026-10-10):
- caches written since the 2026-10-08 merge reference 'AI_fit' (or
  'refine_sampling') — they unpickle wherever this alias layer is
  importable;
- PACKAGE-ERA caches (pre-merge, e.g. cmp_fits_exactera.npz)
  reference 'lorentz_fit.FitResult' and unpickle ONLY where the
  legacy lorentz_fit/ package is importable — i.e. in THIS repo.
  A migrated tree (Zandpack/AI_code/FittingRoutine/) does NOT carry
  lorentz_fit: there, package-era caches FAIL to unpickle by
  design and must be regenerated; the compatibility guarantee
  covers post-merge caches only.

New code should import AI_fit_base directly.
"""
import sys as _sys

import AI_fit_base as _base

_sys.modules[__name__] = _base
