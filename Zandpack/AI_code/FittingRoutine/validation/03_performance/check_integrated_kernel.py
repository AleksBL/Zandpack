"""Integrated-kernel identity: current AI_fit.residual_and_jacobian
(numba path) vs the pre-edit backup (pure numpy) — bitwise."""
import importlib.util
import sys

import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as new  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "AI_fit_pre", ROOT + "/backups/AI_fit_2026-10-09_pre_jac_kernel.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

rng = np.random.default_rng(5)
ok = True
for (N, M, n_ch, ridge) in [(189, 24, 12, 0.0), (189, 50, 12, 0.05),
                            (189, 100, 12, 0.0), (451, 60, 36, 0.05),
                            (100, 8, 5, 0.0)]:
    E = np.sort(rng.uniform(-1, 1, N))
    B = rng.normal(size=(n_ch, N))
    theta = np.empty(2 * M)
    theta[0::2] = rng.uniform(-1, 1, M)
    theta[1::2] = rng.uniform(0.005, 0.1, M)
    sw = np.ones(N)
    ra, Ja = old.residual_and_jacobian(E, B, theta, sw, None, ridge)
    rb, Jb = new.residual_and_jacobian(E, B, theta, sw, None, ridge)
    same = np.array_equal(ra, rb) and np.array_equal(Ja, Jb)
    ok &= same
    print(f"N={N} M={M} n_ch={n_ch} ridge={ridge}: "
          f"{'bitwise identical' if same else 'DIFF'}")
print("INTEGRATED KERNEL:", "PASS" if ok else "FAIL")
