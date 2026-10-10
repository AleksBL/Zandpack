import gzip, os, pickle, time
import numpy as np
os.chdir("/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/Ag3Au3")
with gzip.open("tmp.gz", "rb") as f:      # pickles FIRST (shadow rule)
    R = pickle.load(f)
import sys
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from refine_sampling import extract_dense
from lorentz_fit.varproj import hermitian_channels, pack_responses
for lead in (0, 1):
    E, F = extract_dense(R, lead, 0)
    p = F.shape[1]
    chans = hermitian_channels(p)
    B = pack_responses(F, chans)
    t = time.time()
    s = np.linalg.svd(B, compute_uv=False)
    print(f"lead{lead}: p={p}, n_ch={B.shape[0]}, N={B.shape[1]} "
          f"[svd {time.time()-t:.0f}s]")
    s0 = s[0]
    for frac in (1e-2, 1e-3, 1e-4, 1e-6, 1e-9):
        r = int((s > frac * s0).sum())
        print(f"   rank at {frac:g}x s0: {r} of {B.shape[0]} "
              f"({100*r/B.shape[0]:.0f}%)")
