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

Note: package-era caches ALSO reference lorentz_fit.FitResult, so the
lorentz_fit/ package directory stays in the repo root (importable)
until those caches are regenerated; it is not the active code.
New code should import AI_fit_base directly.
"""
import sys as _sys

import AI_fit_base as _base

_sys.modules[__name__] = _base
