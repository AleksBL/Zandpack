import os, sys
import numpy as np
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, 'Data2', 'nogap_Ag5'))

# lead0 caches from stage-1 sweep
for tag in ['w05', 'w03']:
    src, dst = f'sweep_fits/m1_mx62_{tag}.npz', f'sweep_fits/m1_mx62_{tag}_lead0.npz'
    if os.path.exists(src) and not os.path.exists(dst):
        d = np.load(src, allow_pickle=True)
        np.savez(dst, fit=d['fit'].item())
        print('seeded', dst)

# lead1 w05 = the final production fit
d = np.load('cmp_fits.npz', allow_pickle=True)
dst = 'sweep_fits/m1_mx62_w05_lead1.npz'
if not os.path.exists(dst):
    np.savez(dst, fit=d['lead1_ik0'].item())
    print('seeded', dst)
print('done')
