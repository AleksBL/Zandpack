import os, sys
import numpy as np
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, 'Data2', 'nogap_Ag5'))
f0 = np.load('sweep_fits/m1_mx62_w03_lead0.npz', allow_pickle=True)['fit'].item()
f1 = np.load('sweep_fits/m1_mx62_w03_lead1.npz', allow_pickle=True)['fit'].item()
np.savez('cmp_fits.npz', lead0_ik0=f0, lead1_ik0=f1)
print('w03 adopted: lead0', f0.n_poles, 'poles, lead1', f1.n_poles, 'poles')
