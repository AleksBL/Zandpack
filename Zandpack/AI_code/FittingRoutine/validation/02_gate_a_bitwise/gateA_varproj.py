"""Rigorous Gate A: bit-for-bit identity of the OLD (backup) AI_fit and
the NEW AI_fit (with operator_threshold knob) on identical problems.

Same data, same seeds, same config -> e/w/C arrays must be EXACTLY
equal (np.array_equal) with the default operator_threshold, and the
knob must not leak into results.
"""
import importlib.util
import sys

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as new  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "AI_fit_pre", ROOT + "/backups/AI_fit_2026-10-09_pre_varproj_opt.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def make_case(seed, p=6, M=6, N=120, noise=1e-3):
    rng = np.random.default_rng(seed)
    E = np.linspace(-5, 5, N)
    e_t = rng.uniform(-4, 4, M)
    w_t = rng.uniform(0.05, 0.4, M)
    C = np.zeros((M, p, p), dtype=complex)
    for k in range(M):
        H = rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))
        C[k] = (H + H.conj().T) / 2
    F = old.evaluate(E, e_t, w_t, C)
    F += (rng.normal(size=F.shape) + 1j * rng.normal(size=F.shape)) * noise
    return E, F


CONFIGS = [
    dict(n_init=20, max_poles=40, target_rms=5e-3, psd=True),
    dict(n_init=20, max_poles=40, target_rms=2e-3, psd=True,
         psd_force=False, eta_raw=1e-2, eta_floor_mult=3.0,
         w_max_raw=0.5, local_floor_frac=1.5),
    dict(n_init=16, max_poles=36, target_rms=1e-3, psd=False),
]

all_ok = True
for ci, cfg in enumerate(CONFIGS):
    for seed in [1, 2]:
        E, F = make_case(seed)
        r_old = old.fit(E, F, **cfg)
        r_new = new.fit(E, F, **cfg)
        same = (np.array_equal(r_old.e, r_new.e)
                and np.array_equal(r_old.w, r_new.w)
                and np.array_equal(r_old.C, r_new.C))
        all_ok &= same
        print(f"config {ci} seed {seed}: n {r_old.n_poles}/{r_new.n_poles} "
              f"rms {r_old.rms:.6g}/{r_new.rms:.6g} "
              f"exact={same}")

print("GATE A (module identity):",
      "PASS - bit-for-bit identical" if all_ok else "FAIL")
