import gzip, pickle
import numpy as np
np.set_printoptions(precision=4, suppress=True)

R = pickle.load(gzip.open('/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/nogap_Ag5/tmp.gz', 'rb'))
print("type:", type(R).__name__)
print("num_leads:", R.num_leads)
print("Contour shape:", np.asarray(R.Contour).shape)
si = np.asarray(R.sampling_idx)
E = np.asarray(R.Contour)[R.sampling_idx[0]]
print("E real range:", E.real.min(), E.real.max(), " N =", len(E))
print("eta:", E.imag[0])
d = np.diff(E.real)
print("spacing: min %.4g max %.4g median %.4g" % (d.min(), d.max(), np.median(d)))
dense = d < 0.02
print("dense region(s):", E.real[:-1][dense].min() if dense.any() else None, "to", E.real[:-1][dense].max() if dense.any() else None, " count", dense.sum())
m = E.real < -3
print("below -3: N =", m.sum(), "spacings:", np.unique(np.round(np.diff(E.real[m]), 4)))
G = R.Nonortho_Gammas[0]
v0 = G.vals
print("len vals per lead:", [len(v) for v in v0])
print("shape of vals[0][0]:", np.asarray(v0[0][0]).shape)
for lead in range(R.num_leads):
    A = R.bs2np(R.Nonortho_Gammas[lead])
    print("bs2np shape lead", lead, A.shape)
    for ik in range(A.shape[0]):
        tr = np.array([np.trace(A[ik, k].real) for k in R.sampling_idx[lead]])
        pltE = E.real
        print(f"  lead{lead} ik{ik}: Tr Gamma range {tr.min():.3g}..{tr.max():.3g}")
        m = pltE < -3
        seg = tr[m]
        pk = (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])
        print(f"    below -3: N {m.sum()}, Tr min {seg.min():.3g} max {seg.max():.3g}, local maxima {pk.sum()}")
        m2 = (pltE >= 2) & (pltE <= 4)
        seg2 = tr[m2]
        pk2 = (seg2[1:-1] > seg2[:-2]) & (seg2[1:-1] > seg2[2:])
        print(f"    [2,4]: N {m2.sum()}, Tr min {seg2.min():.3g} max {seg2.max():.3g}, local maxima {pk2.sum()}")
