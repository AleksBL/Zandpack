import sys, time
import numpy as np
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import AI_fit as lf

p = 12
true_e = np.array([-5.2, -3.3, -0.4, 0.0, 1.2, 3.3, 4.1, 6.0])
true_w = np.array([0.05, 0.02, 0.3, 0.15, 0.08, 0.04, 0.25, 0.5])

for seed in [3, 7, 11]:
    rng = np.random.default_rng(seed)
    E = np.linspace(-7, 7, 400)
    Cm = np.zeros((len(true_e), p, p), dtype=complex)
    for k in range(len(true_e)):
        H = rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))
        Cm[k] = (H + H.conj().T) / 2
    F = lf.evaluate(E, true_e, true_w, Cm)
    F += (rng.normal(size=F.shape) + 1j * rng.normal(size=F.shape)) * 3e-3 * np.abs(F).mean()

    fd = lf.prepare(E, F)
    chans = lf.hermitian_channels(p)
    B = lf.pack_responses(fd.F, chans)
    # linspace init: NOT at the solution, 24 pairs (overparameterized)
    e0, w0 = lf.linspace_init(fd.E, 24)
    bounds = lf.default_bounds(fd.E, w_min=0.01 / fd.e_scale,
                               w_max=0.6 / fd.e_scale)
    out = {}
    for method in ['trf', 'dogbox']:
        t = time.time()
        r = lf.fit_poles(fd.E, B, e0, w0, method=method, bounds_fun=bounds)
        dt = time.time() - t
        out[method] = (r['cost'], r['nfev'], dt, r['status'])
        print(f"seed {seed} {method:7s}: cost {r['cost']:.6g}  "
              f"nfev {r['nfev']:4d}  {dt:5.2f}s  status {r['status']}")
    better = min(out, key=lambda m: out[m][0])
    print(f"  -> lower cost: {better}  "
          f"(cost ratio trf/dogbox = {out['trf'][0]/out['dogbox'][0]:.3f})")
