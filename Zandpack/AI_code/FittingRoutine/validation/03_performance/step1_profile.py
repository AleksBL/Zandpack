"""Step 1 (speedupguide_new.md): profile the current RSSE production fit
and split the time into fun-core / jac / scipy-SVD / other."""
import cProfile
import gzip
import lzma
import os
import pickle
import pstats
import sys
import time

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined'))

# pickles first
def load(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)

R = load('C1.xz')

sys.path.insert(0, ROOT)
import AI_fit  # noqa: E402
from refine_sampling import extract_dense, fit_boosted  # noqa: E402

CFG = dict(n_init=24, max_poles=150, target_rms=5e-4,
           psd=True, psd_force=False, eta_raw=1e-2,
           local_floor_frac=1.5, w_max_raw=0.5,
           eta_floor_mult=3.0)

E, F = extract_dense(R, 0, 0)
print(f"N={len(E)}, p={F.shape[1]}", flush=True)

# confirm channel compression active: n_ch the optimizer sees
fd = AI_fit.prepare(E, F)
chans = AI_fit.hermitian_channels(fd.F.shape[1])
B = AI_fit.pack_responses(fd.F, chans)
print(f"n_ch raw={B.shape[0]}, compressed={AI_fit.compress_channels(B).shape[0]}",
      flush=True)

t0 = time.time()
pr = cProfile.Profile()
pr.enable()
r = fit_boosted(E, F, dict(CFG), n_res=8)
pr.disable()
wall = time.time() - t0
print(f"fit: n={r.n_poles} rms={r.rms:.6g} wall={wall:.1f}s "
      f"(cProfile-inflated)", flush=True)

st = pstats.Stats(pr)
total = wall

# category extraction from stats
rows = {}
for func, (cc, nc, tt, ct, callers) in st.stats.items():
    key = f"{func[0]}:{func[1]}({func[2]})"
    rows[key] = (tt, ct)

def cat(patterns):
    return sum(tt for key, (tt, ct) in rows.items()
               if any(p in key for p in patterns))

core = cat(["_varproj_core"])
jac = cat(["_varproj_jac", "_assemble_dense_j", "_jac_dense_kernel",
           "residual_and_jacobian"])
svd = cat(["_svd", "decomp_svd", "svd.py"])
# subtract: svd inside varproj paths already counted? _varproj_core/jac
# do NOT call svd (QR path); fallback lstsq calls are separate
lstsq_fallback = cat(["lstsq"])
other = total - core - jac - svd

print(f"\ncategory split (wall {total:.1f}s):")
print(f"  _varproj_core (fun path)      : {core:6.2f}s "
      f"({100*core/total:4.1f}%)")
print(f"  _varproj_jac + J assembly     : {jac:6.2f}s "
      f"({100*jac/total:4.1f}%)")
print(f"  scipy SVD (trust region)      : {svd:6.2f}s "
      f"({100*svd/total:4.1f}%)")
print(f"  lstsq (fallback etc.)         : {lstsq_fallback:6.2f}s")
print(f"  other                         : {other:6.2f}s "
      f"({100*other/total:4.1f}%)")
print("\n--- top 15 tottime ---")
st.sort_stats("tottime").print_stats(15)
pr.dump_stats("/tmp/rsse_step1.prof")
