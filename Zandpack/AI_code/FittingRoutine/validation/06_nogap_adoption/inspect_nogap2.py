import gzip, pickle
import numpy as np

R = pickle.load(gzip.open('/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/nogap_Ag5/tmp.gz', 'rb'))
E = np.asarray(R.Contour)[R.sampling_idx[0]].real
A = R.bs2np(R.Nonortho_Gammas[0])[0]
tr = np.array([np.trace(A[k].real) for k in R.sampling_idx[0]])
# full profile coarse print
for a, b in [(-8, -6), (-6, -5), (-5, -4.5), (-4.5, -4), (-4, -3.5), (-3.5, -3), (-3, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 10)]:
    m = (E >= a) & (E < b)
    seg = tr[m]
    print(f"[{a:>5},{b:>5}): N={m.sum():4d} min={seg.min():9.4g} max={seg.max():9.4g} med={np.median(seg):9.4g}")
# locate the below -3 spike
m = E < -3
seg = tr[m]
i0 = np.argmax(seg)
print("\nbelow -3 spike: E=%.3f Tr=%.4g" % (E[m][i0], seg[i0]))
lo, hi = max(0, i0-12), min(len(seg), i0+13)
for i in range(lo, hi):
    print(f"  E={E[m][i]:7.3f}  Tr={seg[i]:10.4g}")
# and the [2,4] monster
m2 = (E >= 2) & (E <= 4)
seg2 = tr[m2]
i1 = np.argmax(seg2)
print("\n[2,4] peak: E=%.3f Tr=%.4g" % (E[m2][i1], seg2[i1]))
lo, hi = max(0, i1-8), min(len(seg2), i1+9)
for i in range(lo, hi):
    print(f"  E={E[m2][i]:7.3f}  Tr={seg2[i]:10.4g}")
np.savez('/tmp/nogap_tr.npz', E=E, tr=tr)
