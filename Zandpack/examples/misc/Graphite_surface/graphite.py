import numpy as np
from ase import Atoms
from ase.visualize import view
from siesta_python.siesta_python import SiP
from scipy.optimize import minimize
import sisl
import matplotlib.pyplot as plt
na, nc = 10,10
Em = np.zeros((na, nc))
a0,c0 = 2.5, 3.3
av = np.linspace(a0-0.25, a0+0.25, na)
cv = np.linspace(c0-0.25, c0+0.25, nc)
def func(x):
    a,c = x
    print(x)
    A = np.array([[0.5*a, -0.5*np.sqrt(3)*a, 0],
                  [0.5*a, +0.5*np.sqrt(3)*a, 0],
                  [0    ,     0,             c]])
    ps = np.vstack([1/4 * A[2], 3/4*A[2], 
                   A.T.dot([1/3, 2/3, 1/4]),
                   A.T.dot([2/3, 1/3, 3/4])], )
    clc = SiP(A, ps, np.array([6]*4),
              basis='SZ', kp = [9,9,9], mpi='mpirun ')
    clc.fdf()
    clc.run_siesta_in_dir()
    return  clc.get_potential_energy()
x0 = np.array([2.55087062, 5.52578585])
# res = minimize(func, x0, method='nelder-mead')
# In[]
x0  = np.array([2.55087062, 5.52578585])
a,c = x0
A   = np.array([[0.5*a, -0.5*np.sqrt(3)*a, 0],
                [0.5*a, +0.5*np.sqrt(3)*a, 0],
                [0    ,     0,             c]])
ps = np.vstack([1/4 * A[2], 3/4*A[2], 
               A.T.dot([1/3, 2/3, 1/4]),
               A.T.dot([2/3, 1/3, 3/4])], )
clc = SiP(A, ps, np.array([6,6,6,6]),
          basis='SZ', kp = [9,9,9], mpi='mpirun ', 
          TwoDim=False, Standardize_cell=True)
clc.fdf()
clc.run_siesta_in_dir()
p, n = clc.make_bandpath()
H    = clc.read_TSHS()
nk   = 400
band = sisl.BandStructure(H, p, nk, n)
# In[]
eigs = np.zeros((nk, H.no))
lk, kt, kl = band.lineark(True)
plt.xticks(kt, kl)
for i,k in enumerate(band.k):
    eigs[i] = H.eigh(k=k)
plt.plot(lk, eigs, color='k'); plt.ylim(-4,4)
plt.title('Bulk Graphite')
plt.savefig('Bands2.svg')

# for i in range(na):
#     for j in range(nc):
#         a = av[i] # 2.5
#         c = cv[j] # 3.3
#         A = np.array([[0.5*a, -0.5*np.sqrt(3)*a, 0],
#                       [0.5*a, +0.5*np.sqrt(3)*a, 0],
#                       [0    ,     0,             c]])
        
#         ps = np.vstack([1/4 * A[2], 3/4*A[2], 
#                        A.T.dot([1/3, 2/3, 1/4]),
#                        A.T.dot([2/3, 1/3, 3/4])], )
#         clc = SiP(A, ps, np.array([6,6,6,6]),
#                   basis='SZ', kp = [5,5,5], mpi='mpirun ')
#         clc.fdf()
#         clc.run_siesta_in_dir()
#         Em[i,j] = clc.get_potential_energy()
        
#         #A = Atoms(positions=ps, numbers=[6,6,6,6], cell=A)
#         #view(A)
