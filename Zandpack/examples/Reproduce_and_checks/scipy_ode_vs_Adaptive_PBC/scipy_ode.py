from TimedependentTransport.TimedependentTransport import scipy_ode
from TimedependentTransport.TimedependentTransport import AdaptiveRK4 as RK4

from TimedependentTransport.Pulses import zero_dH, zero_bias
import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from shared import dH, delta



C = load(open('Graw.Timedep','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices()

C.Check_input_to_ODE()
f     = C.make_f_experimental()
sig   = C.sigma
psi   = C.Psi_vec
omega = C.omegas

t,dm,jl =  scipy_ode(f, sig, psi, omega, 0.0, np.linspace(0,20,200)+1e-5, dH, delta, C.Ixi, 
                      dH_given = True, method = 'RK45', dt_guess = None, 
                      atol = 1e-7, rtol = 1e-5)

plt.show()
plt.plot(t, jl)
plt.savefig('scipy_ode.png', dpi = 300)
