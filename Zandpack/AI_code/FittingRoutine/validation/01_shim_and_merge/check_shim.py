import os, sys
import numpy as np
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
sys.path.insert(0, ROOT)
import refine_sampling as rs
import AI_fit

# shim re-exports AI_fit objects?
checks = {
    'extract_dense': rs.extract_dense is AI_fit.extract_dense,
    'inject_fits': rs.inject_fits is AI_fit.inject_fits,
    'fit_boosted': rs.fit_boosted is AI_fit.fit_boosted,
    'CombinedFit': rs.CombinedFit is AI_fit.CombinedFit,
    'FitResult': rs.FitResult is AI_fit.FitResult,
    'refine_loop': rs.refine_loop is AI_fit.refine_loop,
    'resample': rs.resample is AI_fit.resample,
    'sampled_transmission_dos': rs.sampled_transmission_dos is AI_fit.sampled_transmission_dos,
    '_fit_one': rs._fit_one is AI_fit._fit_one,
}
for k, v in checks.items():
    print(k, 'OK' if v else 'MISMATCH')
assert all(checks.values())

# old cache unpickles through the shim (needs lorentz_fit for FitResult)
os.chdir(os.path.join(ROOT, 'Data2', 'nogap_Ag5'))
d = np.load('cmp_fits.npz', allow_pickle=True)
f0 = d['lead0_ik0'].item()
print('cache unpickle OK:', type(f0).__name__, f0.n_poles, 'poles')
f1 = d['lead1_ik0'].item()
print('cache unpickle OK lead1:', f1.n_poles, 'poles')
print('SHIM VALID')
