import lzma, os, pickle, sys
import numpy as np
ROOT = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine'
os.chdir(os.path.join(ROOT, 'Data2', 'VerySmallRSSE', 'refined'))
sys.path.insert(0, ROOT)
import AI_fit as lf
sys.path.remove(ROOT)

def load(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)

R = load('C1.xz')
sys.path.insert(0, ROOT)
from refine_sampling import extract_dense
sys.path.remove(ROOT)
E_raw, F_raw = extract_dense(R, 0, 0)
fd = lf.prepare(E_raw, F_raw)
chans = lf.hermitian_channels(fd.F.shape[1])
B = lf.pack_responses(fd.F, chans)
print('B shape (n_ch, N):', B.shape)

# what production uses: compress_channels defaults (rank_tol=1e-9, min_gain=0.75)
Bc = lf.compress_channels(B)
print('compress_channels default ->', Bc.shape)

# rank at various tols
sv = np.linalg.svd(B, compute_uv=False)
for tol in [1e-6, 1e-8, 1e-9, 1e-10, 1e-12]:
    print(f'  rank at {tol:g}: {int((sv > tol * sv[0]).sum())}')

# effective J shape in production at M=50 with compressed B
M = 50
rng = np.random.default_rng(0)
e = rng.uniform(-1, 1, M); w = rng.uniform(0.01, 0.1, M)
theta = np.empty(2*M); theta[0::2], theta[1::2] = e, w
sw = np.ones(fd.E.size)
import time
r, J = lf.residual_and_jacobian(fd.E, Bc, theta, sw, None, 0.0)
print('production J shape (N*r, 2M):', J.shape)
t0 = time.perf_counter()
for _ in range(20):
    lf.residual_and_jacobian(fd.E, Bc, theta, sw, None, 0.0)
print(f'resid+jac (compressed, M=50): {(time.perf_counter()-t0)/20*1000:.2f} ms')
t0 = time.perf_counter()
for _ in range(20):
    np.linalg.svd(J, full_matrices=False)
print(f'scipy-style dense SVD of J {J.shape}: '
      f'{(time.perf_counter()-t0)/20*1000:.2f} ms')
