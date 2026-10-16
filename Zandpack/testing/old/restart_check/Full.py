from TimedependentTransport.TimedependentTransport import AdaptiveRK4  as RK4
from TimedependentTransport.Pulses import zero_dH, zero_bias
import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from shared import dH, delta


C = load(open('Graw.Timedep','rb'))
C.diagonalise()
C.get_propagation_quantities()
C.get_dense_matrices_purenp()
f     = C.make_f_purenp()
sig   = C.sigma
psi   = C.Psi_vec
omega = C.omegas

# teq, deq = RK4( f, sig, psi, omega, 1e-5,  -0, 30,  zero_dH, zero_bias, C.Ixi, 
                # 0.1, fixed_mode = False, name = 'Chain')
tb, db   = RK4( f, sig, psi, omega, 1e-5,  -0, 10,  dH, delta, C.Ixi, 
                0.1, fixed_mode = False, name = 'Chain')

plt.plot(tb, np.array(db['current_left']))
plt.savefig('Jl_AdaptiveRK4_full.png', dpi = 300)
