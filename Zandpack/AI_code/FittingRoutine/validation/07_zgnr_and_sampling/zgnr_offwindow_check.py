"""Off-window / between-sample sanity check for the zgnr refined fits.

Reloads tmp_refined.gz (the refined sampling), refits both leads, and
checks the padded-window behavior (max excursion, total-fit min
eigenvalue) the same way bench/plot_examples.py does.
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

with gzip.open(os.path.join(ROOT, "Data2", "zgnr", "tmp_refined.gz"),
               "rb") as f:
    R = pickle.load(f)

sys.path.insert(0, ROOT)
from lorentz_fit import fit as lfit  # noqa: E402
from lorentz_fit import validate as val  # noqa: E402

for lead in range(R.num_leads):
    E, F = ex.extract_dense(R, lead, 0)
    t0 = time.time()
    res = lfit(E, F, n_init=24, max_poles=60, target_rms=2e-3,
               channel_tol=1e-4)
    rms = val.frobenius_rms_error(res.evaluate(E), F)
    span = E.max() - E.min()
    pad = 0.30 * span
    eg = np.linspace(E.min() - pad, E.max() + pad, 1001)
    Fe = res.evaluate(eg)
    me_fit = np.linalg.eigvalsh(Fe).min(axis=1)
    in_w = (eg >= E.min()) & (eg <= E.max())
    # off-window excursion relative to the data scale
    scale = np.linalg.norm(F, axis=(1, 2)).max()
    off = np.abs(Fe[~in_w]).max()
    d = np.diff(np.sort(E))
    print(f"zgnr refined lead{lead}: N={len(E)} poles={res.n_poles} "
          f"rms={rms:.4g} | off-window max |F|={off:.4g} "
          f"(data scale {scale:.4g}, ratio {off/scale:.3g}) | "
          f"min eig in-window={me_fit[in_w].min():+.3g} "
          f"full-grid={me_fit.min():+.3g} "
          f"max spacing={d.max():.3g} "
          f"[{time.time()-t0:.0f}s]", flush=True)
