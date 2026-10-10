"""PSD-residue variant for the zgnr refined grid: does psd=True cure the
between-sample lambda_min dips, and what does it cost in rms?"""
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
               channel_tol=1e-4, psd=True)
    rms = val.frobenius_rms_error(res.evaluate(E), F)
    eg = np.linspace(E.min(), E.max(), 1001)
    me = np.linalg.eigvalsh(res.evaluate(eg)).min(axis=1)
    print(f"zgnr refined lead{lead} PSD: poles={res.n_poles} "
          f"rms={rms:.4g} applied={res.psd['applied'] if res.psd else None} "
          f"min eig={me.min():+.3g} [{time.time()-t0:.0f}s]", flush=True)
