"""Diagnose the skrf-init recovery failure: does the relocation lose a
true pole, or does the optimizer fail from a good init?"""
import sys

sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
import numpy as np

from lorentz_fit import fit
from lorentz_fit import varproj as vp
from lorentz_fit.data import prepare
from lorentz_fit.poles_init import init_poles
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/tests")
from test_synthetic import make_synthetic  # noqa: E402

rng = 0
E = np.linspace(-5.0, 5.0, 401)
M, p = 12, 4
centers, widths, C, F, _ = make_synthetic(M, p, E, rng=rng)
print("true poles (sorted by center):")
for c, w in sorted(zip(centers, widths)):
    print(f"  e={c:+.4f}  w={w:.4f}")

fd = prepare(E, F)
init = init_poles(fd.E, fd.F, M, method="skrf", e_margin=0.25)
print(f"\nskrf init returned {init[0].size} poles:")
for c, w in sorted(zip(init[0], init[1])):
    print(f"  e={c:+.4f}  w={w:.4f}")

res = fit(E, F, n_poles=M, init="skrf", tail_anchors=False, e_margin=0.25,
          ridge=0.0, w_min=2.5e-3, verbose=True)
print(f"\nfit: n={res.n_poles} rms={res.rms:.4g} history={res.history}")
