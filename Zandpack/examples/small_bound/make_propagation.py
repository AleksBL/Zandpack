#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:43:38 2022

@author: aleksander
"""

from TimedependentTransport.Optimized_RK45 import AdaptiveRK4_Opti as RK4
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from TimedependentTransport.Pulses import zero_dH, zero_bias

Test = load(open('Testfit.Timedep','rb'))

Test.diagonalise()
Test.get_propagation_quantities()
Test.get_dense_matrices_purenp(zero_tol = 1e-5)
Test.Check_input_to_ODE()
f = Test.make_f_purenp()
Test.write_to_file('SmallBound_file')
assert 1 == 0

# In[]
# Test.Device.kp_tbtrans = np.array([0.0, 0.0, 0.0])
# E_QP1, vec1, v11, v21 = Test.Device.solve_qp_equation( .2 , np.array([0., 0., 0.]), its = 20)
# E_QP2, vec2, v12, v22 = Test.Device.solve_qp_equation(-.2 , np.array([0., 0., 0.]), its = 20)
# E_QP3, vec3, v13, v23 = Test.Device.solve_qp_equation(+.3 , np.array([0., 0., 0.]), its = 20)
# E_QP4, vec4, v14, v24 = Test.Device.solve_qp_equation(2.1 , np.array([0., 0., 0.]), its = 20)
# E_QP5, vec5, v15, v25 = Test.Device.solve_qp_equation(-.5 , np.array([0., 0., 0.]), its = 20)

#vec = np.load('QP_mode.npy')

# # In[]
# def log(x):
#     return np.log10(x)

# def project_on_vec1():
#     return log(vec1.conj().dot(d0['density matrix']).dot(vec1))
# def project_on_vec2():
#     return log(vec2.conj().dot(d0['density matrix']).dot(vec2))
# def project_on_vec3():
#     return log(vec3.conj().dot(d0['density matrix']).dot(vec3))
# def project_on_vec4():
#     return log(vec4.conj().dot(d0['density matrix']).dot(vec4))
# def project_on_vec5():
#     return log(vec5.conj().dot(d0['density matrix']).dot(vec5))

# In[]
from TimedependentTransport.Pulses import zero_dH, zero_bias

sig = Test.sigma
psi = Test.Psi_vec
omg = Test.omegas

_t0, _d0 = RK4(f, sig, psi, omg, 1e-5, 0, 100, zero_dH, zero_bias, Test.Ixi, name = 'GB')
plt.plot(_t0, _d0['current_left'])
plt.show()

# In[]
from TimedependentTransport.Pulses import box_pulse
from TimedependentTransport.Pulses import air_photonics_pulse as AP

ramp = -np.linspace(-.5, .5, 10)
def dH(t,sig):
    dh = np.zeros(sig.shape, dtype = np.complex128)
    dh[0,8:18,8:18] += AP(t)  * np.diag(ramp)
    dh[0,18:, 18:]  -= AP(t)  * np.diag(np.ones(8))
    dh[0,0:8, 0:8]  += AP(t)  * np.diag(np.ones(8))
    return dh

def bias(t,a):
    if t<0.0:
        return 0.0
    else:
        if a==0:
            return  AP(t)
        if a==1:
            return -AP(t)

t0, d0 = RK4(f, None, None, None, 1e-7, 0, 1000, dH, bias, Test.Ixi, name = 'GB')

 # In[]
piv = Test.pivot
xyz = Test.Device.pos_real_space[piv]
idxu = [i for i in range(len(xyz)) if np.mod(xyz[i,0],1)>0.1 ]
idxd = [i for i in range(len(xyz)) if np.mod(xyz[i,0],1)<0.1 ]
Nu = np.trace(d0['density matrix'][:,0,idxu,:][:,:,idxu], axis1 = 1, axis2 = 2)
Nd = np.trace(d0['density matrix'][:,0,idxd,:][:,:,idxd], axis1 = 1, axis2 = 2)

plt.plot(t0, Nd, label = 'up')
plt.plot(t0, Nu, label = 'down')

plt.legend()
plt.show()
plt.plot(t0, d0['current_left'])
plt.plot(t0, d0['current_right'])


# In[]
r = 1
xyz = Test.Device.pos_real_space[Test.Device.tbtrans_params_dic['pivot']]
X = np.zeros((len(xyz), len(xyz)))
for i in range(len(xyz)):
    for j in range(len(xyz)):
        if i==j:
            X[i,j] = 0.#xyz[i,0]
        elif np.linalg.norm(xyz[i] - xyz[j]) <0.6:
            X[i,j] = r

dip = np.trace(d0['density matrix']@X, axis1 = 2, axis2 = 3)
N   = np.trace(d0['density matrix'],   axis1 = 2, axis2 = 3)


