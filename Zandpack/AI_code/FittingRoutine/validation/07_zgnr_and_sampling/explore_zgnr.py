"""Explore zgnr tmp.gz: grid, block structure, per-lead Gamma samples."""
import gzip
import os
import pickle
import sys

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, os.path.join(ROOT, "bench"))
import extract_data as ex  # noqa: E402  (loads before lorentz_fit import)

import numpy as np  # noqa: E402

path = os.path.join(ROOT, "Data2", "zgnr", "tmp.gz")
with gzip.open(path, "rb") as f:
    R = pickle.load(f)

print("type:", type(R).__name__)
print("num_leads:", R.num_leads)
print("sampling_idx:", [np.asarray(s).shape for s in R.sampling_idx])
for lead in range(R.num_leads):
    sidx = np.asarray(R.sampling_idx[lead])
    E = np.asarray(R.Contour[sidx]).real
    print(f"\nlead {lead}: N={len(sidx)}, E in [{E.min():.3f}, {E.max():.3f}]")
    d = np.diff(E)
    print(f"  uniform? median dE={np.median(d):.5g}  min={d.min():.5g}  "
          f"max={d.max():.5g}")
    G = R.Nonortho_Gammas[lead]
    print("  block inds:", G.inds)
    for bi, (I, J) in enumerate(G.inds):
        print(f"    block ({I},{J}) vals shape {np.asarray(G.vals[bi]).shape}")
    E2, F = ex.extract_dense(R, lead, 0)
    tr = np.real(np.trace(F, axis1=1, axis2=2))
    print(f"  dense F: {F.shape}, trace range [{tr.min():.4g}, {tr.max():.4g}]")
    # how sharp is the function? biggest jump between adjacent samples
    dtr = np.abs(np.diff(tr))
    print(f"  max |d trace| between adjacent samples: {dtr.max():.4g} "
          f"at E={E[1:][np.argmax(dtr)]:.3f}")
    # effective rank of the matrix function
    ev = np.linalg.eigvalsh(F)
    print(f"  eig range: min={ev.min():.4g} max={ev.max():.4g}")
