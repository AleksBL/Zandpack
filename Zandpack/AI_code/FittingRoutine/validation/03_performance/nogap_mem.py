"""Step 2 (speedupguide_new.md): nogap fit peak-RSS, A1 vs pre-A1.

Usage: python3 nogap_mem.py <current|backups/AI_fit_2026-10-09_pre_A1_split.py>
Prints wall time; peak RSS captured externally via /usr/bin/time -v.
"""
import gzip
import importlib.util
import os
import pickle
import resource
import sys
import threading
import time

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'nogap_Ag5'))


def load(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


R = load('tmp.gz')   # pickles first

which = sys.argv[1]
if which == 'current':
    sys.path.insert(0, ROOT)
    import AI_fit as mod
    sys.path.remove(ROOT)
else:
    spec = importlib.util.spec_from_file_location(
        "AI_fit_old", os.path.join(ROOT, which))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_old"] = mod
    spec.loader.exec_module(mod)
    mod._njit = None if False else mod._njit  # keep its own path

sys.path.insert(0, ROOT)
from refine_sampling import extract_dense  # noqa: E402
sys.path.remove(ROOT)

E, F = extract_dense(R, 0, 0)
print(f"module={which} N={len(E)} p={F.shape[1]}", flush=True)

CFG = dict(n_init=24, max_poles=62, target_rms=2e-3,
           psd=True, psd_force=False, eta_raw=2e-2,
           local_floor_frac=1.5, w_max_raw=0.3,
           eta_floor_mult=1.0, channel_tol=1e-4)

rss_log = []
stop = threading.Event()


def poll():
    while not stop.is_set():
        try:
            with open('/proc/self/statm') as f:
                rss_log.append(int(f.read().split()[1]) * 4096 / 1e6)
        except Exception:
            pass
        stop.wait(2.0)


threading.Thread(target=poll, daemon=True).start()

t0 = time.time()
r = mod.fit(E, F, **CFG)
stop.set()
print(f"RESULT {which}: n={r.n_poles} rms={r.rms:.6g} "
      f"wall={time.time() - t0:.1f}s", flush=True)
print(f"RSS curve (MB, ~2s samples, n={len(rss_log)}): "
      f"{[round(x) for x in rss_log]}", flush=True)
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB",
      flush=True)
