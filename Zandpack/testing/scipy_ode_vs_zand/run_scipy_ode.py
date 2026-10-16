
# In[]
# from Zandpack.Optimized_RK45 import AdaptiveRK4_Opti as RK4
from Zandpack.decoupled_functions import scipy_ode

# from Zandpack.Pulses import zero_dH, zero_bias
import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from shared import dH, delta
from time import time
import lzma

C = load(lzma.open('Graw.xz','rb'))
#C.diagonalise()
#C.get_propagation_quantities()
#C.get_dense_matrices_purenp()
#f     = C.make_f_purenp()
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices(1e-8)
C.Check_input_to_ODE(loose_fermi = True)
f     = C.make_f()
sig   = C.sigma
psi   = C.Psi_vec
omega = C.omegas
#tb, db   = RK4( f, sig, psi, omega, 1e-5,  -0, 20,  dH, delta, C.Ixi, 
#                0.1, fixed_mode = False, name = 'Chain')

t,dm,jl =  scipy_ode(f, sig, psi, omega, 0.0, np.linspace(0,20,200)+1e-5, dH, delta, C.Ixi, 
                      dH_given = True, method = 'RK45', dt_guess = None, 
                      atol = 1e-8, rtol = 1e-8)
plt.show()
plt.plot(t, jl)
plt.savefig('scipy_ode.png', dpi = 300)


# plt.plot(tb, np.array(db['current_left']))
# plt.savefig('Jl_AdaptiveRK4_opti.png', dpi = 300)
