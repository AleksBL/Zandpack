import sys
import numpy as np
sys.path.insert(0, "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine")
from lorentz_fit.sampling import suggest_insertions_derivative

# sharp flank at E=0.3 + smooth bump at -2: steep part must get points
E = np.linspace(-5, 5, 41)          # coarse, spacing 0.25
w = np.array([0.05, 0.5])
c = np.array([0.3, -2.0])
Phi = (w[None, :] / ((E[:, None] - c[None, :]) ** 2 + w[None, :] ** 2)) / np.pi
F = np.zeros((E.size, 2, 2))
F[:, 0, 0] = Phi @ np.array([1.0, 2.0])
F[:, 1, 1] = Phi @ np.array([0.5, 1.5])
F[:, 0, 1] = Phi @ np.array([0.2, 0.3]); F[:, 1, 0] = F[:, 0, 1]

ins = suggest_insertions_derivative(E, F, tol=3.0, max_add=8)
print("insertions:", np.round(ins, 3))
assert ins.size > 0, "no proposals!"
near_flank = np.abs(ins - 0.3) < 0.5
assert near_flank.sum() >= 2, "flank not prioritized!"
# tol high -> nothing
ins2 = suggest_insertions_derivative(E, F, tol=1e3)
assert ins2.size == 0
print("PASS: flank prioritized, tol gate works")
