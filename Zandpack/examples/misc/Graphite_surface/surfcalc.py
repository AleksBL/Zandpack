import numpy as np
from ase import Atoms
from ase.visualize import view
from siesta_python.siesta_python import SiP
from scipy.optimize import minimize
import sisl
import matplotlib.pyplot as plt
from siesta_python.funcs import pyinds2siesta

basis = 'SZP'
a,c = [2.55087062, 5.52578585]
A = np.array([[0.5*a, -0.5*np.sqrt(3)*a, 0],
              [0.5*a, +0.5*np.sqrt(3)*a, 0],
              [0    ,     0,             c]])
ps = np.vstack([1/4 * A[2], 
                3/4*A[2], 
                A.T.dot([1/3, 2/3, 1/4]),
                A.T.dot([2/3, 1/3, 3/4])])
ps += A[2]
blk = SiP(A, ps, np.array([6,6,6,6]),
          directory_name = 'bulk',
          basis=basis, kp = [9,9,9], mpi='mpirun ', 
          TwoDim=False, semi_inf = '-a3', sl = 'bulk',
          Standardize_cell=True)
blk.fdf(parallel_k = True); blk.run_siesta_in_dir()
gs = blk.to_sisl().move(-A[2]).tile(5, 2).add_vacuum(15, 2)
surf = SiP(gs.cell, gs.xyz, gs.atoms.Z,
          directory_name = 'surf',
          basis=basis, kp = [9,9,1], mpi='mpirun ', 
          elecs = [blk], Chem_Pot = [0,],
          solution_method='transiesta')
def buffer_cond(r):
    if r[2] < blk.pos_real_space[:,2].min() - 1e-2:
        return True
    return False

surf.find_elec_inds()
surf.set_buffer_atoms(buffer_cond)
idx_r, idx_f = [], []
for ia in range(len(surf.s)):
    if surf.pos_real_space[ia,2]<11.0:#if ia in surf.buffer_atoms or ia in surf.elec_inds[0]:
        idx_f += [ia]
    else:
        idx_r += [ia]

fc = pyinds2siesta(np.array(idx_f))
rc = pyinds2siesta(np.array(idx_r))
surf.fdf_relax(['atom ' +fc, 'clear '+rc])
surf.fdf(parallel_k=True)
surf.write_more_fdf(['%include MD.fdf'], name='RUN')

surf.run_siesta_in_dir()

# In[]
from SelfEnergyCalculators.Decimation import BTDinv, find_pivotting

nk = 300
ne = 100
Eg = np.linspace(-.2,.2, ne) + 1e-2j
# p,n = surf.make_bandpath()
np.savez('info.npz', buffer_atoms = surf.buffer_atoms, p=p, n=n)
Hb = sisl.get_sile("PATH/TO/elecH.TSHS").read_hamiltonian()# blk.read_TSHS()
HS = sisl.get_sile("PATH/TO/surfH.TSHS").read_hamiltonian()
Hs = Hs.remove(surf.buffer_atoms)
SE = sisl.physics.RecursiveSI(Hb, '-C')
# Put in the points on the surface youre interested in
band = sisl.BandStructure(Hs, [[1/3-1/12, 1/3-1/12 , 0],
                               [1/3,       1/3-1   , 0]
                               ], nk, ["P1", "P2"])

DOS = np.zeros((nk, len(Eg)), dtype=complex)
elec_o = []
for i in range(len(Hb.xyz)):
    dist    = np.linalg.norm(Hb.xyz[i] - Hs.xyz, axis=1)
    idx     = np.where(dist < 1e-3)[0][0]
    elec_o += [Hs.a2o(idx, all=True)]
elec_o = np.hstack(elec_o)
# elec_o = np.hstack([np.arange(Hs.a2o(ie), Hs.a2o(ie+1)) for ie in surf.elec_inds[0]])
# def func(hs, ss,ssdense, SE, k, E, p):
#     iG = ss * E - hs
#     iG[elec_o[:, None], elec_o[None, :]] -= SE.self_energy(E, k = k)
for i,k in enumerate(band.k):
    hs = Hs.Hk(k = k, dtype=np.complex128)#.toarray()
    ss = Hs.Sk(k = k, dtype=np.complex128)#.toarray()
    ssdense = ss.toarray()
    for ie in range(ne):
        E  = Eg[ie]
        iG = ss * E - hs
        iG[elec_o[:, None], elec_o[None, :]] -= SE.self_energy(E, k = k)
        #if ie == 0:
        #    p,part,btd,ip = find_pivotting(iG, start = 18)
        G = np.linalg.inv(iG.toarray())#G = BTDinv(iG.toarray(), p, btd, ip, restrict = 2)
        DOS [i, ie] = 1j*np.sum(ssdense*(G - G.conj().T))
        
