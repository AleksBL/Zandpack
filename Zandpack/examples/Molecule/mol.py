import sisl
import matplotlib.pyplot as plt
import numpy as np
from TimedependentTransport import TD_Transport, AdaptiveRK4_Htnojit, AdaptiveRK4, hbar
from numba import njit
from funcs import diff_central, calc_current
from ase import Atoms
from ase.build import molecule
from ase.visualize import view

#Define sampling of the lead Gammas
eps  = (1e-6 + 1j  * 1e-2)
line = np.linspace(-8, 8, 152) + eps
# line = np.array([-8, -4, -2.5, -1.0,  1.0, 2.0 ,3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) + eps
line = np.vstack((line,line))
# make a molecule
BDA = molecule('BDA')
BDA.cell = np.diag(1.44 * np.ones(3))
# BDA.edit()
BDA = sisl.Geometry.fromASE(BDA)
# make leads
move_up = 5.0 #+ 1.44 * 2 
move_dw = -14.8# -1.44 * 2
L  = sisl.geom.sc(1.44, 'C').tile(8,1)
LU = L.move([0,move_up,0]).add_vacuum(10,0).add_vacuum(10,2)
LD = L.move([0,move_dw,0]).add_vacuum(10,0).add_vacuum(10,2)
dev = BDA.add(LU).add(LD)
dev = dev.add_vacuum(20,0).add_vacuum(dev.xyz[:,1].max() - dev.xyz[:,1].min(),1).add_vacuum(20,2)
dev = dev.move(dev.cell[1,:]/2)
LU  = sisl.geom.sc(1.44, 'C').tile(4,1).move([0,move_up,0]).add_vacuum(10,0).add_vacuum(10,2)
LD  = sisl.geom.sc(1.44, 'C').tile(4,1).move([0,move_dw,0]).add_vacuum(10,0).add_vacuum(10,2)
LU  = LU.move(LU.cell[1,:] )#+ np.array([0,    2 * 1.44, 0]))
LD  = LD.move(-LD.cell[1,:]/2 + np.array([0, 2 * 1.44, 0]))
LU  = LU.move (dev.cell[1,:]/2)
LD  = LD.move (dev.cell[1,:]/2)

plt.show()
sisl.plot(dev); plt.axis('equal')
# In[]
Test = TD_Transport([LD,LU], dev, kT_i = [0.025, 0.025])
Test.Make_Contour(line, 20, pole_mode = 'JieHu2011')
plt.show()
Test.Electrodes( semi_infs = ['-a2', '+a2'] , kp = [[1,50,1], [1,50,1]])
Test.make_device()
plt.show()
Test.Device.Visualise()
plt.show()

# In[]
Test.run_electrodes()
Test.run_device()

# In[]

sub_indices = []
TBT = sisl.get_sile(Test.Device.dir + '/siesta.TBT.nc')
count = 0
atoms = [TBT.geom.atoms[i] for i in range(TBT.geom.na) if i in TBT.a_dev]
for ia, a in enumerate(atoms):
    for io in range(TBT.geom.orbitals[ia]):
        if io == 2:
            sub_indices += [count]
        count+=1

# In[]
Test.read_data(fact = 2.0, NumL=10, #Fallback_W= 20,
                    fit_mode = 'all', use_analytical_jac = True, 
                    maxiter = 10 ** 6,min_method = 'L-BFGS-B', 
                    ebounds = (line.min().real-1 , line.max().real+1 ), 
                    wbounds = (0.01, 5.0),
                    
                    tol = -1.0, options = {'disp':True, 'maxiter': 10**6, 'gtol': 0.001},
                    sub_orbital = sub_indices, # Not really using sub_orbital from sisl, but the idea is the same I guess
                    fit_real_part  = True
                    )

plt.show()
Test.get_propagation_quantities(); Test.get_dense_matrices(zero_tol = 1e-12)


Vi = np.linspace(-.4,.4,20); idx = np.argsort(np.abs(Vi)); Vi = Vi[idx]
# Test.run_device_non_eq(Vi)
# h,v = Test.neq_Hs()
# assert 1 == 0
sig = Test.sigma
psi = Test.Psi_vec
omega = Test.omegas
no_d = sig.shape[2]

# In[] Timepropagation
V = 1.0
@njit
def Bias(t, tp, ts):
    return V * (np.tanh(t / ts) - np.tanh((t - tp) / ts)) / 2
@njit
def fd(x, mu, s):
    return 1 / (1 + np.exp((x - mu) / s))
@njit
def delta(t, a):
    if a == 0:
        return   Bias(t, 20, 1)/2
    elif a == 1:
        return - Bias(t, 20, 1)/2

@njit
def zero_bias(t, a):
    return 0.0

@njit
def zero_dH(t, sigma):
    A = np.zeros((1, no_d, no_d), dtype=np.complex128)
    return A

def ht(t, sigma,d1,d2):
    return Test.Hdense[:,0,:,:]

@njit
def dH(t, sigma):
    A = np.zeros((1, no_d, no_d), dtype=np.complex128)
    return A

f   = Test.make_f_general_v2(parallel = True, fastmath = True)
F   = Test.make_f_general(parallel = True, fastmath = True)
xi  = Test.xi
Ixi = Test.Ixi

t1, data1 = AdaptiveRK4(
                        F,
                        sig, psi, omega,
                        1e-5, -15.0, 10.0,
                        zero_dH,
                        zero_bias,
                        Test.Ixi,
                        0.01,
                        fixed_mode=False,
                        name="Multithread_RK4",
)
assert 1 == 0


t2, data2 = AdaptiveRK4_Htnojit(
                                f,
                                sig, psi, omega,
                                1e-5, -15.0, 10.0,
                                ht,
                                zero_bias,
                                Test.Ixi,
                                0.01,
                                fixed_mode=False,
                                name="Multithread_RK4",
)
assert 1 == 0

sig          = data1['last sigma']
omega        = data1['last omega']
psi          = data1['last psi']

t3, data3 = AdaptiveRK4(
                        f,
                        sig, psi, omega,
                        1e-5, -5.0, 40.0,
                        dH,
                        delta,
                        Test.Ixi,
                        0.01,
                        fixed_mode=False,
                        name="Multithread_RK4",
)

dNdt = diff_central(t3, np.array([np.trace(data3['density matrix'][i]) for i in range(len(t1))]))
JL   = data3['current left']
JR   = data3['current right']
