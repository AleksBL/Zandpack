"""Reference fits on the ORIGINAL zgnr grids (both leads, full 251 pts)."""
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
    R = pickle.load(f)

sys.path.insert(0, ROOT)
from lorentz_fit import fit as lfit  # noqa: E402
from lorentz_fit import validate as val  # noqa: E402

for lead in range(R.num_leads):
    E, F = ex.extract_dense(R, lead, 0)
    t0 = time.time()
    res = lfit(E, F, n_init=24, max_poles=60, target_rms=2e-3,
               channel_tol=1e-4)
    smp = val.frobenius_rms_error(res.evaluate(E), F)
    print(f"zgnr lead{lead} FULL-GRID reference: N={len(E)} p={F.shape[1]} "
          f"n_poles={res.n_poles} sample rms={smp:.4g} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    np.savez(f"/tmp/zgnr_full_ref_lead{lead}.npz", e=res.e, w=res.w)
