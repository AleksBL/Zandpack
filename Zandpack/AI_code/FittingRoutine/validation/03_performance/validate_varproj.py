"""Validate the _varproj_factors in-place rewrite:
1. unit-level bitwise identity (QR path + forced rank-deficient
   fallback path) vs the pre-edit backup module
2. per-call timing at RSSE production shapes
"""
import importlib.util
import sys
import time

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as new  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "AI_fit_pre", ROOT + "/backups/AI_fit_2026-10-08_pre_channelcomp.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

rng = np.random.default_rng(11)

print("=== unit-level bitwise identity ===")
all_ok = True
for N, M, n_ch in [(189, 24, 12), (189, 50, 12), (189, 100, 12),
                   (451, 60, 36), (100, 8, 5)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    # QR path (generic theta) + ridge path
    for ridge in [0.0, 0.05]:
        ra, xa, za, ka, sa = old._varproj_factors(E, B, theta, sw, None, ridge)
        rb, xb, zb, kb, sb = new._varproj_factors(E, B, theta, sw, None, ridge)
        same = all(np.array_equal(a, b) for a, b in
                   [(ra, rb), (xa, xb), (za, zb), (ka, kb), (sa, sb)])
        all_ok &= same
        print(f"  N={N} M={M} n_ch={n_ch} ridge={ridge}: "
              f"{'bitwise identical' if same else 'DIFF'}")
    # fallback path: duplicate poles -> rank-deficient design
    theta[2] = theta[0]; theta[3] = theta[1]
    ra, xa, za, ka, sa = old._varproj_factors(E, B, theta, sw, None, 0.0)
    rb, xb, zb, kb, sb = new._varproj_factors(E, B, theta, sw, None, 0.0)
    same = all(np.array_equal(a, b) for a, b in
               [(ra, rb), (xa, xb), (za, zb), (ka, kb), (sa, sb)])
    all_ok &= same
    print(f"  N={N} M={M} n_ch={n_ch} fallback(dup poles): "
          f"{'bitwise identical' if same else 'DIFF'}")

print("UNIT:", "PASS" if all_ok else "FAIL")

print("\n=== per-call timing (min of 200, ms) ===")
print(f"{'shape':>22s} {'old':>8s} {'new':>8s} {'speedup':>8s}")
for N, M, n_ch in [(189, 24, 12), (189, 50, 12), (189, 100, 12)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    ts = {}
    for tag, mod in [("old", old), ("new", new)]:
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(200):
                mod._varproj_factors(E, B, theta, sw, None, 0.05)
            best = min(best, (time.perf_counter() - t0) / 200)
        ts[tag] = best * 1000
    print(f"  N={N} M={M:3d} r={n_ch:2d}   {ts['old']:7.3f} {ts['new']:7.3f} "
          f"{ts['old'] / ts['new']:7.2f}x")
