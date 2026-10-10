"""Shared harness for the perturbed-start basin-spread experiment
(project_handoff.md, follow-up point 3).

Question: the rms gap between tr_solver='gram' and 'exact' measured on
Ag3Au3 (0.0228 vs 0.0149) could be (a) gram-specific (weak-direction
truncation noise) or (b) ordinary adaptive-loop basin sensitivity that
'exact' itself has. Single-start comparisons cannot tell. This round
measures the START-TO-START DISTRIBUTIONS: jitter the init poles by
~1e-6 eV, refit with each solver, and compare the outcome
distributions (rms / n poles / min eig / wall / T rms).

Standing rules: OMP_NUM_THREADS=1; load pickles BEFORE any repo-root
sys.path insert; the repo-root Zandpack/Block_matrices folders are
inspection copies and must never shadow the installed packages.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'

# solver variants: label -> fit kwargs (everything else identical)
SOLVERS = [
    ("exact",        dict(tr_solver="exact")),
    ("gram_cut1e-7", dict(tr_solver="gram", gram_cut=1e-7)),
    ("gram_cut1e-9", dict(tr_solver="gram", gram_cut=1e-9)),
    ("gram_cut0",    dict(tr_solver="gram", gram_cut=0.0)),
]

N_PERT = 6
SEED = 20261009
JITTER_RAW = 1e-6  # eV, raw units


def import_ai_fit():
    """Import AI_fit with the repo root temporarily on sys.path."""
    sys.path.insert(0, ROOT)
    import AI_fit as lf
    sys.path.remove(ROOT)
    return lf


def make_starts(n_poles, e_scale, n_pert=N_PERT, seed=SEED,
                jitter_raw=JITTER_RAW):
    """Starts for one system: c0 = control (no jitter, must reproduce
    the un-seeded baseline exactly), s1..sN have every init-pole center
    AND width jittered by U(-jitter, jitter) eV in raw units, converted
    to the scaled units fit(init_poles=...) consumes."""
    rng = np.random.default_rng(seed)
    starts = [("c0", 0.0, 0.0)]
    for s in range(1, n_pert + 1):
        de = rng.uniform(-jitter_raw, jitter_raw, n_poles) / e_scale
        dw = rng.uniform(-jitter_raw, jitter_raw, n_poles) / e_scale
        starts.append((f"s{s}", de, dw))
    return starts


def baseline_init_poles(lf, E, F, n_init=24, e_margin=0.15):
    """The deterministic init pole set the production fit uses (psd=True
    -> the NNLS PSD dictionary; same dispatch, same inputs as _fit_impl
    would compute internally). Passing these back via init_poles=c0/w0
    (unjittered) reproduces the un-seeded baseline exactly."""
    fd = lf.prepare(E, F)
    c0, w0 = lf._poles_init_dispatch("skrf", fd.E, fd.F, n_init,
                                     e_margin=e_margin, psd=True)
    return c0, w0, fd


def gamma_metrics(fit, E_line, pad=2.0, npts=801):
    Eg = np.linspace(E_line.min() - pad, E_line.max() + pad, npts)
    me = float(np.linalg.eigvalsh(fit.evaluate(Eg)).min())
    return dict(n_poles=int(fit.n_poles), rms=float(fit.rms),
                boost_parts=list(fit.history.get("boost_parts", [])),
                min_eig=me)


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def done_keys(path):
    """(solver, start) pairs already in the JSONL - resume support."""
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["solver"], r["start"]))
            except Exception:
                pass
    return keys


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
