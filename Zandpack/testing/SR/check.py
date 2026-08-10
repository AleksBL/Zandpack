from pickle import load
import numpy as np
from lzma import open
from Zandpack.TimedependentTransport import AdaptiveRK4  as RK4
from Zandpack.TimedependentTransport import scipy_ode
from numba import njit
# In[]
C = load(open('Fit.xz','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(zero_tol=1e-8)
C.Check_input_to_ODE()
f     = C.make_f()
sig,psi,omega   = C.sigma,C.Psi_vec,C.omegas
@njit
def bias(t,a):
    ex = np.exp(-(t- 4*np.pi))
    dec = min((ex,1))
    f = (1/(2.0+np.sin(t + np.sin(t)**2)))*dec-0.5
    if a == 0:
        return + 0.05/((t-75)**2 + 0.025)
    if a == 1:
        return - 0.05/((t-75)**2 + 0.025)
@njit
def dH(t,sig):
    o  = np.zeros(sig.shape, dtype=np.complex128)
    o[0,4,4] += sig[0,4,4] - sig[0,3,3]/3
    o[0,3,3] -= sig[0,3,3] + sig[0,4,4]/3
    return o
assert 1 == 0
ts, ds   =  RK4(f, sig,psi, omega, 1e-5, 0, 20, dH, bias, C.Ixi, name='stm')
t1,dm,jl =  scipy_ode(f, sig, psi, omega, 0.0, np.linspace(0,20,50)+1e-5, dH, bias, C.Ixi, 
                      dH_given = True, method = 'RK45', dt_guess = None, 
                      atol = 1e-6, rtol = 1e-3)

