"""Layout check: does bs2np's dense layout match extract_dense's sorted
block layout? (zgnr, zgnr_spin)"""
import gzip
import os
import pickle
import sys

import numpy as np

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, os.path.join(ROOT, "bench"))
import extract_data as ex  # noqa: E402

for tag, path in (
        ("zgnr", os.path.join(ROOT, "Data2", "zgnr", "tmp.gz")),
        ("zgnr_spin", os.path.join(ROOT, "Data2", "zgnr_spin", "tmp.gz"))):
    with gzip.open(path, "rb") as f:
        R = pickle.load(f)
    for lead in range(R.num_leads):
        sidx = R.sampling_idx[lead]
        E, F_mine = ex.extract_dense(R, lead, 0)
        F_b2n = np.asarray(R.bs2np(R.Nonortho_Gammas[lead]))[0][sidx]
        F_b2n = 0.5 * (F_b2n + np.conj(np.swapaxes(F_b2n, -1, -2)))
        dev = np.abs(F_mine - F_b2n).max()
        print(f"{tag} lead{lead}: extract_dense vs bs2np max dev "
              f"{dev:.4g} (scale {np.abs(F_mine).max():.4g}) "
              f"{'MATCH' if dev < 1e-8 else 'MISMATCH'}")
