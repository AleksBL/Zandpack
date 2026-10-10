import numpy as np
import os, sys
sys.path.insert(0, '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine')
os.chdir('/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2/nogap_Ag5')
os.makedirs('sweep_fits', exist_ok=True)
d = np.load('cmp_fits.npz', allow_pickle=True)
np.savez('sweep_fits/m1_mx62_w05.npz', fit=d['lead0_ik0'].item())
print('seeded baseline')
