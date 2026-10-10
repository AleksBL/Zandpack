import gzip, os, pickle, sys, time
import numpy as np
os.chdir("/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/Ag3Au3")
with gzip.open("tmp.gz", "rb") as f:
    R = pickle.load(f)
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from refine_sampling import extract_dense, fit_boosted
E, F = extract_dense(R, 1, 0)
print(f"[time] ag3au3 lead1: {len(E)} pts, p={F.shape[1]}", flush=True)
t = time.time()
res = fit_boosted(
    E, F,
    dict(n_init=24, max_poles=100, target_rms=2e-3,
         psd=True, psd_force=False, eta_raw=1e-2,
         local_floor_frac=1.5, w_max_raw=0.5,
         eta_floor_mult=5.0, channel_tol=1e-4),
    n_res=8)
print(f"[time] NEW code: n={res.n_poles} relrms={res.rms:.4g} "
      f"in {time.time()-t:.0f}s", flush=True)
