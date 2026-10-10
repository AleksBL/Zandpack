"""Offline validation of sampling refinement on zgnr lead 0.

The 251-point uniform sampling is the ground truth. We
  A) coarsen it 4x, fit, and measure the error vs truth,
  B) propose insertions from the coarsened samples, look up their values
     in the truth (simulating a TBtrans evaluation), refit, re-measure,
  C) run a second refine round on B,
  D) run removal on the full 251-point grid and refit what remains,
  E) unit-check the Lagrange weights.

Load order: the TD_Transport pickle MUST be loaded before the repo root
goes on sys.path (the Zandpack excerpt folder shadows the installed
package).
"""
import gzip
import os
import pickle
import sys
import time

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, os.path.join(ROOT, "bench"))
import extract_data as ex  # noqa: E402

import numpy as np  # noqa: E402

# --- load truth BEFORE any repo-root sys.path insert ----------------------
path = os.path.join(ROOT, "Data2", "zgnr", "tmp.gz")
with gzip.open(path, "rb") as f:
    R = pickle.load(f)
E_full, F_full = ex.extract_dense(R, 0, 0)
print(f"zgnr lead0: {F_full.shape} on [{E_full.min()}, {E_full.max()}]")

sys.path.insert(0, ROOT)
from lorentz_fit import fit as lfit  # noqa: E402
from lorentz_fit import sampling as samp  # noqa: E402
from lorentz_fit import validate as val  # noqa: E402

FIT_KW = dict(n_init=24, max_poles=60, target_rms=2e-3, channel_tol=1e-4)


def fit_eval(E, F, tag):
    t0 = time.time()
    res = lfit(E, F, **FIT_KW)
    Ffit = res.evaluate(E_full)
    err = val.frobenius_rms_error(Ffit, F_full)
    smp = val.frobenius_rms_error(res.evaluate(E), F)
    dt = time.time() - t0
    print(f"{tag}: N={len(E):4d}  n_poles={res.n_poles:3d}  "
          f"sample rms={smp:.4g}  TRUTH rms={err:.4g}  [{dt:.0f}s]")
    return res, err


def lookup(Eq, E_all, F_all):
    """Nearest-index lookup of F at energies Eq from the dense truth."""
    idx = np.searchsorted(E_all, Eq)
    idx = np.clip(idx, 1, len(E_all) - 1)
    pick = np.where(np.abs(E_all[idx] - Eq) < np.abs(E_all[idx - 1] - Eq),
                    idx, idx - 1)
    return F_all[pick]


# --- E) Lagrange weights unit check --------------------------------------
x = np.linspace(0.0, 1.7, 4)
w = samp._lagrange_weights(x, 0.63)
y = 3.2 - 1.5 * x + 0.7 * x ** 2 - 0.4 * x ** 3
est = np.dot(w, y)
assert abs(est - (3.2 - 1.5 * 0.63 + 0.7 * 0.63 ** 2 - 0.4 * 0.63 ** 3)) < 1e-12
assert abs(w.sum() - 1.0) < 1e-12
print("[PASS] Lagrange weights reproduce a cubic exactly; sum(w)=1")

# --- A) coarsen 4x --------------------------------------------------------
step = 4
E_c, F_c = E_full[::step], F_full[::step]
res_c, err_c = fit_eval(E_c, F_c, "A coarse x4     ")

# --- B) one insertion round from the coarse samples ----------------------
ins = samp.suggest_insertions(E_c, F_c, tol=5e-3,
                              poles=(res_c.e, res_c.w))
print(f"\nB: {len(ins)} insertion(s) proposed: "
      f"{np.round(ins, 3)[:12]}{' ...' if len(ins) > 12 else ''}")

F_ins = lookup(ins, E_full, F_full)
order = np.argsort(np.concatenate([E_c, ins]))
E_b = np.sort(np.concatenate([E_c, ins]))
F_b = np.concatenate([F_c, F_ins])[order]
res_b, err_b = fit_eval(E_b, F_b, "B coarse+insert")
print(f"   truth error improvement: {err_c:.4g} -> {err_b:.4g} "
      f"({err_b / err_c:.2f}x)")

# --- C) a second refinement round on B ------------------------------------
ins2 = samp.suggest_insertions(E_b, F_b, tol=5e-3,
                               poles=(res_b.e, res_b.w))
print(f"\nC round2: {len(ins2)} more insertion(s)")
if len(ins2):
    F_ins2 = lookup(ins2, E_full, F_full)
    order = np.argsort(np.concatenate([E_b, ins2]))
    E_c2 = np.sort(np.concatenate([E_b, ins2]))
    F_c2 = np.concatenate([F_b, F_ins2])[order]
    res_c2, err_c2 = fit_eval(E_c2, F_c2, "C round2 fit   ")
    print(f"   truth error improvement: {err_b:.4g} -> {err_c2:.4g} "
          f"({err_c2 / err_b:.2f}x)")

# --- D) removal on the full grid ------------------------------------------
rem = samp.suggest_removals(E_full, F_full, tol=1e-2)
keep = np.setdiff1d(np.arange(len(E_full)), rem)
d = np.diff(E_full[keep])
print(f"\nD removal: {len(rem)} of {len(E_full)} points removed; "
      f"kept {len(keep)}, max kept gap {d.max():.4g} "
      f"(median {np.median(d):.4g})")
removed_near = E_full[rem][np.argsort(np.abs(E_full[rem]))][:6]
print(f"   closest-to-zero removed energies: {np.round(removed_near, 3)}")
E_d, F_d = E_full[keep], F_full[keep]
res_d, err_d = fit_eval(E_d, F_d, "D thinned      ")
