"""Build AI_fit.py: merge lorentz_fit/*.py + refine_sampling.py into a
single module (migration plan step 1). Mechanical: relative imports
dropped, module-alias prefixes stripped, everything at top level."""
import re

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
ORDER = [
    ("data",            f"{ROOT}/lorentz_fit/data.py"),
    ("varproj",         f"{ROOT}/lorentz_fit/varproj.py"),
    ("poles_init",      f"{ROOT}/lorentz_fit/poles_init.py"),
    ("prune",           f"{ROOT}/lorentz_fit/prune.py"),
    ("constraints",     f"{ROOT}/lorentz_fit/constraints.py"),
    ("validate",        f"{ROOT}/lorentz_fit/validate.py"),
    ("sampling",        f"{ROOT}/lorentz_fit/sampling.py"),
    ("pipeline",        f"{ROOT}/lorentz_fit/__init__.py"),
    ("TD glue",         f"{ROOT}/refine_sampling.py"),
]

REL_IMPORT = re.compile(r"^\s*from \.")
FUTURE = re.compile(r"^\s*from __future__")


def strip_relative(lines):
    out, skip = [], False
    for ln in lines:
        if skip:
            # continuation of a dropped statement?
            skip = ln.rstrip().endswith("\\")
            continue
        if REL_IMPORT.match(ln) or FUTURE.match(ln):
            skip = ln.rstrip().endswith("\\")
            continue
        out.append(ln)
    return out


PREFIXES = ["_lf.varproj.", "_lf.validate.", "_lf.",
            "_vp.", "_prune.", "_con.", "_val.", "_data.", "_samp."]


def strip_prefixes(text):
    for p in PREFIXES:
        text = text.replace(p, "")
    return text


parts = [f'''"""AI_fit - shared-pole Lorentzian fit of Hermitian matrix
functions + adaptive level-width sampling refinement, single-file
edition (merged from the lorentz_fit package + refine_sampling.py;
see the project HANDOFF for the full documentation and the migration
plan for the Zandpack context).

QUICK START (replaces the PoleGuess -> R.Fit -> curvefit_all ->
iterative_PSD -> FitNO2O block of a 3.py script):

    fits = {{}}
    for lead in range(R.num_leads):
        for ik in range(len(R.Nonortho_Gammas[0].vals[0])):
            E, F = extract_dense(R, lead, ik)
            fits[(lead, ik)] = fit_boosted(
                E, F,
                dict(n_init=24, max_poles=100, target_rms=2e-3,
                     psd=True, psd_force=True,
                     eta_raw=1e-2,        # contour imaginary part
                     eta_floor_mult=5.0,  # per-system! see table below
                     w_max_raw=0.5, local_floor_frac=1.5,
                     channel_tol=1e-4 if F.shape[1] > 32 else 0.0),
                n_res=8)
    inject_fits(R, fits)          # setLval + FitNO2O internally
    C = find_correction(R, Emin=-3, Emax=3.0)   # your existing tail
    R.Renormalise_H(C)

TUNED KNOBS PER SYSTEM (measured; eta_floor_mult is the sharp one):
    zgnr            mult 5.0  w_max 0.35 psd_force True
    zgnr_spin       mult 5.0  w_max 0.5  psd_force True  (0.35 regresses)
    VerySmallRSSE   mult 3.0  w_max 0.5  psd_force False (E=0 notch)
    agnr            mult 1.0  w_max 0.5  psd_force False (mult 5 kills it)
    spinpol_ag5     mult 2.0  w_max 0.5  psd_force False
    Fig2Ag5         mult 1.0  w_max 0.5  psd_force False
    Ag3Au3(-gap)    mult 5.0  w_max 0.5  psd_force False
    AuBreakJunction mult 2.0  w_max 0.5  psd_force False
Keep OMP_NUM_THREADS=1 (measured faster AND reproducible).
Sections below mirror the original modules (data, varproj, poles_init,
prune, constraints, validate, sampling, pipeline, TD glue).
"""
import gzip
import lzma
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import find_peaks
from scipy.sparse.linalg import LinearOperator

__all__ = ["fit", "FitResult", "prepare", "FitData", "fit_boosted",
           "CombinedFit", "inject_fits", "extract_dense", "resample",
           "refine_loop", "sampled_transmission_dos", "load_pickle",
           "current_lines", "suggest_insertions", "suggest_removals",
           "suggest_insertions_derivative", "validate",
           "suggest_pole_insertions"]
''']

for name, path in ORDER:
    lines = open(path).read().splitlines(keepends=True)
    lines = strip_relative(lines)
    text = "".join(lines)
    if name == "TD glue":
        # remove the explicit-path importlib machinery FIRST (raw text:
        # the removal regex anchors on the un-stripped lines), then
        # remap the _fit() calls to the top-level fit()
        text = re.sub(
            r"def _import_lorentz_fit\(\):.*?_fit = _lf\.fit\n",
            "", text, flags=re.S)
        text = text.replace("import importlib.util\n", "")
        text = text.replace("_fit(", "fit(")
        # the _samp/_val aliases loaded the submodules by name; in the
        # single file everything is top level -> drop the assignments
        text = re.sub(r"_samp = importlib\.import_module.*\n", "", text)
        text = re.sub(r"_val = importlib\.import_module.*\n", "", text)
    if name == "prune":
        # name collision with sampling.suggest_insertions (a DIFFERENT
        # concept: residual-peak POLE insertion, not grid proposals):
        # rename prune's variant in the merged file
        text = text.replace("def suggest_insertions(",
                            "def suggest_pole_insertions(")
    if name in ("pipeline", "TD glue"):
        if name == "pipeline":
            text = text.replace("_prune.suggest_insertions",
                                "suggest_pole_insertions")
        text = strip_prefixes(text)
    banner = (f"\n\n"
              f"# {'='*74}\n"
              f"# SECTION: {name}\n"
              f"# {'='*74}\n")
    parts.append(banner + text)

out = "".join(parts)
# collapse >2 consecutive blank lines
out = re.sub(r"\n{4,}", "\n\n\n", out)
with open(f"{ROOT}/AI_fit.py", "w") as f:
    f.write(out)
print(f"AI_fit.py written: {len(out.splitlines())} lines")
