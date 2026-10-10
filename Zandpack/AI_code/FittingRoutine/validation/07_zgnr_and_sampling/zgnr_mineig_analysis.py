"""Where does the zgnr refined-grid fit's lambda_min dip, and does the
full 251-point grid fit have the same issue? Also: how well is the E~0
spike resolved by the refined grid vs the original?"""
import gzip
import os
import pickle
import sys
import time

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, os.path.join(ROOT, "bench"))
import extract_data as ex  # noqa: E402

import numpy as np  # noqa: E402

with gzip.open(os.path.join(ROOT, "Data2", "zgnr", "tmp.gz"), "rb") as f:
    R0 = pickle.load(f)
with gzip.open(os.path.join(ROOT, "Data2", "zgnr", "tmp_refined.gz"),
               "rb") as f:
    RR = pickle.load(f)

sys.path.insert(0, ROOT)
from lorentz_fit import fit as lfit  # noqa: E402
from lorentz_fit import validate as val  # noqa: E402

# spike resolution: points near E=0 on each grid
for tag, R in (("original", R0), ("refined", RR)):
    E = np.asarray(R.Contour[R.sampling_idx[0]]).real
    near = E[np.abs(E) < 0.25]
    tr_pts = np.asarray(R.Contour[R.sampling_idx[0]]).real
    print(f"{tag} grid: N={len(E)}; pts within |E|<0.25: {len(near)} "
          f"({np.round(near, 4)})")

# data shape near 0
E0, F0 = ex.extract_dense(R0, 0, 0)
tr = np.real(np.trace(F0, axis1=1, axis2=2))
sel = np.abs(E0) < 0.3
print("original trace near E=0:")
for e, t in zip(E0[sel], tr[sel]):
    print(f"  E={e:+.4f}  Tr={t:9.3f}")

# refined-grid fit: locate the lambda_min dip
E, F = ex.extract_dense(RR, 0, 0)
t0 = time.time()
res = lfit(E, F, n_init=24, max_poles=60, target_rms=2e-3, channel_tol=1e-4)
eg = np.linspace(E.min(), E.max(), 2001)
Fe = res.evaluate(eg)
me = np.linalg.eigvalsh(Fe).min(axis=1)
i = int(np.argmin(me))
print(f"\nrefined lead0 fit ({time.time()-t0:.0f}s): rms="
      f"{val.frobenius_rms_error(res.evaluate(E), F):.4g}, "
      f"lambda_min dip {me[i]:+.3g} at E={eg[i]:+.4f} "
      f"(nearest refined sample {E[np.argmin(np.abs(E - eg[i]))]:+.4f})")
in_sp = (np.abs(eg) < 0.5)
print(f"  lambda_min in |E|<0.5: {me[in_sp].min():+.3g}; "
      f"outside: {me[~in_sp].min():+.3g}")

# full-grid fit for comparison
t0 = time.time()
res2 = lfit(E0, F0, n_init=24, max_poles=60, target_rms=2e-3,
            channel_tol=1e-4)
eg2 = np.linspace(E0.min(), E0.max(), 2001)
me2 = np.linalg.eigvalsh(res2.evaluate(eg2)).min(axis=1)
j = int(np.argmin(me2))
print(f"full-grid lead0 fit ({time.time()-t0:.0f}s): rms="
      f"{val.frobenius_rms_error(res2.evaluate(E0), F0):.4g}, "
      f"lambda_min dip {me2[j]:+.3g} at E={eg2[j]:+.4f}")
print(f"  lambda_min in |E|<0.5: {me2[np.abs(eg2) < 0.5].min():+.3g}; "
      f"outside: {me2[np.abs(eg2) >= 0.5].min():+.3g}")
