import os
T0 = float(os.environ.get('ZP_T0', -147.12))   # fast mode, see run_zandpack_tests --fast
T1 = float(os.environ.get('ZP_T1', 150))
# This test's window is ~297 time units, i.e. ~15x longer than any other test's.
# scipy_ode returns the current only at these points, so the shared 200-point
# default would sample it at dt = 1.5 and the interpolated comparison would be
# dominated by that (measured: max|dJ| 1.92e-02 at 200 points vs 1.33e-03 at
# 2000). The device is tiny (4 orbitals), so the denser grid is free.
N  = int(os.environ.get('ZP_NEVAL', 2000))
ATOL = float(os.environ.get('ZP_ATOL', 1e-8))  # fast mode loosens these
RTOL = float(os.environ.get('ZP_RTOL', 1e-8))

import numpy as np
import matplotlib.pyplot as plt
import sisl
from numba import njit
from Zandpack.TimedependentTransport import TD_Transport
from Zandpack.Pulses import box_pulse, zero_dH
from Zandpack.TimedependentTransport import hbar as print_hbar
from Zandpack.Pulses import air_photonics_pulse as P
from Zandpack.decoupled_functions import AdaptiveRK4
RK4=AdaptiveRK4
print('Make sure you have put hbar = 1.0 in this module\n')
print('Yes you have to do this manually\n')
print('Go to the location of the package -> td_constants and set hbar = 1.0 for natural units \n\n')
print('hbar is currently set to ',print_hbar)

tx     = 2
t_dev  = 2
t_elec = 20

lat_const = 2.5
lineWB = np.linspace(0.0, 0.0, 1) + 1j * 1e-2
lineNL = np.linspace(-15 , 15, 400)  + 1j*1e-2

line = lineWB
line = np.vstack((line,line))

geom_dev   = sisl.geom.sc(lat_const, 'H').tile(tx+2, 0)
geom_ep    = sisl.geom.sc(lat_const, 'H').tile(1, 0).move(np.array([lat_const * (tx+1), 0, 0]))
geom_em    = sisl.geom.sc(lat_const, 'H').tile(1, 0)

geom_dev   = geom_dev.add_vacuum(10,1).add_vacuum(10,2).add_vacuum(10,0)
geom_em    = geom_em. add_vacuum(10,1).add_vacuum(10,2)
geom_ep    = geom_ep. add_vacuum(10,1).add_vacuum(10,2)

Test = TD_Transport([geom_em,geom_ep], geom_dev, kT_i = [0.1, 0.1], mu_i = [0.0, 0.0])
Test.Make_Contour(line, 15 , pole_mode = 'JieHu2011')

Test.Electrodes( semi_infs = ['-a1', '+a1'] )
Test.make_device(
            elec_inds = [[0],[tx+1]],
            Print= False)

elec = sisl.Hamiltonian(sisl.geom.sc(lat_const, sisl.Atom(1, R= 3.0)).add_vacuum(10,1).add_vacuum(10,2))
elec.construct([[0.1, lat_const * 1.1], 
                [0  , t_elec             ]])

Test.run_electrodes(manual_H = [elec, elec])

dev_H = sisl.Hamiltonian(sisl.geom.sc(lat_const, sisl.Atom(1, R= 3.0)).tile(tx+2,0).add_vacuum(10,1).add_vacuum(10,2).add_vacuum(10,0))
dev_H.construct([[0.1, lat_const * 1.1], 
                  [0,   1              ] ])

#tbtrans cant do 1 orbital, so a (almost) noninteracting one is put in.
dev_H[0,1     ]    =  5**0.5
dev_H[1,0     ]    =  5**0.5
dev_H[0,0] = 1e-10
dev_H[1,1] = 2e-10
dev_H[2,2] = 3e-10
dev_H[3,3] = 4e-10

dev_H[1,2] = 1e-8#0.0
dev_H[2,1] = 1e-8#0.0
dev_H[2,3] = 1e-8#0.0
dev_H[3,2] = 1e-8#0.0
dev_H[tx+1, tx-1]  =  5**0.5
dev_H[tx-1, tx+1]  =  5**0.5
#dev_H[tx+1, tx]  =  5**0.5
#dev_H[tx, tx+1]  =  5**0.5


Test.run_device(manual_H = dev_H, )
Test.read_data()

# In[read stuff]

NL = 1
min_tol  = -0.0*np.ones(NL)
min_tol1 =  min_tol.copy()
min_tol2 =  min_tol.copy()

def run_mini(its):
    Test.Fit(fact = 1.0, Fallback_W = 10.0, NumL = NL,
          fit_mode      = 'all',
          force_PSD     = True,
          force_PSD_tol = [min_tol1, min_tol2],
          use_analytical_jac = True,
          min_method = 'SLSQP',
          ebounds = (-5, 5),
          wbounds = (0.01, 0.8),
          gbounds = (None, None),
          tol = -1,
          options = {'disp':True,'maxiter':its, 
                     'gtol':1e-10,
                     'ftol':1e-10,
                     'iprint':1
                     },
          fit_real_part = False,
          specific_bounds = None,#[{(0 ,2) :[(0.1, 0.11), (4,5)]}, {(0 ,5) :[(-0.1, 0.1), (4,5)]}], 
          alpha_PO = 0.01, 
          )

run_mini(1)

Test.diagonalise()
Test.get_propagation_quantities()
Test.get_dense_matrices(1e-8)

# Write the state the MPI codes read, then load it straight back: that is what
# guarantees both sides start from the same (sigma, xi, Ixi), and it also picks
# up any refinement written into Arrays/ by SCF/psinought (see run_steady.sh).
Test.tofile(name = 'TDT', tol = 1e-8)
from Zandpack.Loader import flexload
A          = 'TDT/Arrays/'
Test.sigma = flexload(A + 'DM_Ortho.npy')
Test.xi    = flexload(A + 'xi.npz')
Test.Ixi   = flexload(A + 'Ixi.npz')
Test.get_propagation_quantities()   # rebuild psi/omega from xi/Ixi

Test.Check_input_to_ODE(loose_fermi = True)
f     = Test.make_f()
sig   = Test.sigma
psi   = Test.Psi_vec
omega = Test.omegas

# In[]

Vmax = 1.0
T = 0.1
W = 0.1
Tmax=10*np.pi
zeta = Tmax
@njit
def pulse(t):
    return np.cos(W*t)#*np.exp(-1/2*(t/zeta)**2)
@njit
def dH(t,sigma):
    A = np.zeros(sigma.shape, dtype = np.complex128)
    A[0,1,1] = 1 * pulse(t)
    return A
@njit
def delta(t,a):
    if  a == 0:
        return 0.0 * pulse(t) 
    if  a == 1:
        return 0.125 * pulse(t)

t, d = RK4( f, sig, psi, omega, 1e-12,  T0, T1,  dH, delta, Test.Ixi, 0.1, fixed_mode = False, 
                    name = 'QD')

# The scipy_ode propagator, on exactly the same f / start state / dH / delta, so
# the methods form a three-way comparison.  This test used to have AdaptiveRK4
# on the Python side only, which is why its cross-method difference stood out
# (see handoff.md).
# shortlib lives at the TestRUN root, whatever the depth of this file
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != '/' and not _os.path.isdir(_os.path.join(_root, 'shortlib')):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from shortlib.driver import propagate          # ZP_RHS / ZP_CHUNK

t1, dm, jl = propagate(Test, dH, delta, T0, np.linspace(T0, T1, N) + 1e-5, ATOL, RTOL)

# In[]
plt.plot(t,  np.array(d['current_left']), label = 'AdaptiveRK4 (left)')
plt.plot(t1, jl.sum(axis = 1),            label = 'scipy_ode (left)')
plt.ylabel('Current left')
plt.xlabel('time')
plt.legend()
plt.savefig("HoneyChurch2018_Jl.svg")
plt.close()

np.savez_compressed("honeychurch2018_results.npz",
                    rk4_jl   = d['current_left'],
                    rk4_jr   = d['current_right'],
                    rk4_t    = t,
                    scipy_t  = t1,
                    scipy_jl = jl,
                    scipy_N  = np.trace(dm, axis1 = -2, axis2 = -1).real)

