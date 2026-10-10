"""Smoke test: fit_poles tr_solver='gram' vs 'exact' vs 'lsmr' on a
small synthetic problem (never_form_J guide section 8 step 2)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf
sys.path.remove(ROOT)

rng = np.random.default_rng(42)
N, M, n_ch = 120, 12, 5
E = np.linspace(-1, 1, N)
e_true = rng.uniform(-0.85, 0.85, M)
w_true = rng.uniform(0.03, 0.25, M)
Phi = lf.lorentzian_basis(E, e_true, w_true)
Xt = rng.normal(size=(M, n_ch))
B = Xt.T @ Phi.T + 0.02 * rng.normal(size=(n_ch, N))
sw = 1.0 / np.linspace(0.5, 1.5, N)
e0, w0 = lf.linspace_init(E, M + 4)
bf = lf.default_bounds(E)

for tag, kw in (("exact", {}), ("lsmr", {}), ("gram", {})):
    t0 = os.times()
    import time; t0 = time.time()
    r = lf.fit_poles(E, B, e0, w0, sample_weights=sw, bounds_fun=bf,
                     ridge=0.05, tr_solver=tag, **kw)
    t1 = time.time()
    print(f"{tag:6s} cost {r['cost']:.10e}  nfev {r['nfev']:4d}  "
          f"status {r['status']}  wall {t1-t0:6.2f}s  msg: {r['message']}")
