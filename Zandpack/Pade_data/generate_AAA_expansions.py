#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:23:59 2026

@author: aleks
"""
import numpy as np
import baryrat
from Zandpack.PadeDecomp import FD_expanded_v2,FD_expanded_v2_opt, FD_expanded, Hu_poles
import matplotlib.pyplot as plt
tab  = np.load('JieHu2011_accuracy_table.npz')
kT   = 1.0
npoles = 20
pot    = -4
def f(x): return 1/(1 + np.exp(x)) - 1/2

def improve(nx, pot):
    idx = np.where(tab['pots'] == pot)[0][0]
    X   = tab['table'][idx]
    idx = np.where(tab['nl'] == nx )[0][0]
    X   = X[idx]
    return X

xph, Rph = Hu_poles(npoles)
Xmax = improve(npoles, pot)
fact = 2.0
xlim = (-fact * Xmax, fact * Xmax)
# xlim = (-500, 500)
# xv   = np.linspace(-1.5 * Xmax, 1.5 * Xmax, 10000)
r  = baryrat.brasil(f, xlim, deg = 2 * npoles)
# r  = baryrat.aaa(xv, f, tol = 1e-7,)
# offset = r(-100000.0) - f(-10000)

xp_T, Rp_T = r.polres()

xv   = np.linspace(-3 * Xmax, 3 * Xmax, 10000)
diff1 = f(xv) - (FD_expanded(xv, xph, 1.0, coeffs = Rph) - 0.5)
# diff1 = (FD_expanded(xv, xph, 1.0, coeffs = Rph) - 0.5)
plt.plot(xv, np.abs(diff1), label = 'JieHu')
# diff2 = f(xv) - (FD_expanded_v2(xv, xph, 1.0, coeffs = Rph) - 0.5)
# plt.plot(xv, np.real(diff2), label = 'JieHu')
plt.legend()
# assert 1 == 0
# In[]
idx    = np.where(xp_T.imag>=1e-10)[0]
xp     = xp_T[idx]
Rp     = Rp_T[idx] 
diff2 = f(xv) - (FD_expanded(      xv, xp, 1/kT, coeffs = -Rp) - 1/2+ r.gain())
# diff2 = (FD_expanded(      xv, xp, 1/kT, coeffs = -Rp) - 1/2) 
diff3 = f(xv) - (FD_expanded_v2(   xv, xp, 1/kT, coeffs = -Rp) - 1/2 + r.gain())


# diff3 = (FD_expanded_v2(   xv, xp, 1/kT, coeffs = -Rp) - 1/2) 
#diff4 = f(xv) - r(xv)

#diff4 = f(xv) - (FD_expanded_v3(   xv, xp_T, 1/kT, coeffs = -Rp_T) - 1/2)

# diff3 = f(xv) - (FD_expanded_v2(xv, xp, 1/kT, coeffs = -Rp) - 1/2)
plt.plot(xv, np.abs(diff2), label = 'brasil FDE' ,       linestyle='dashed')
# plt.plot(xv, diff2.imag, label = 'brasil FDE' ,       linestyle='dashed')
plt.plot(xv, np.abs(diff3), label = 'brasil FDE v2' ,    linestyle='dashed')
# plt.plot(xv, diff3.imag, label = 'brasil FDE v2' ,    linestyle='dashed')
# plt.plot(xv, np.real(diff4), label = 'rfunc' ,    linestyle='dashed')
# plt.plot(xv, diff4.imag, label = 'brasil FDE v3' ,    linestyle='dashed')
plt.legend()
# plt.plot(xv, diff3.real, label = 'brasil FDE v2' , linestyle='dashed')
# plt.legend()
# assert 1 == 0
# # xlin = np.linspace(-1.5 * X , 1.5 * X, 10000)
# # plt.plot(xlin, FD_expanded(xlin, xph, beta = 1.0, coeffs = Rph))
# # plt.show()
#    print(X)
def poleexp(r, x):
    xp, rp= r.polres()
    if (xp.imag == 0).any():
        rp = rp[xp.imag!=0]
        xp = xp[xp.imag!=0]
        
    out = rp[:, None]/(x[None, :] - xp[:, None] )
    return out.sum(axis=0)

#def error(*x):
#    nx = len(x)
#    
#    FD_expanded(      xv, xp, 1/kT, coeffs = -Rp) - 1/2
# plt.show()
# plt.plot(xv, r(xv).real); plt.plot(xv, (poleexp(r, xv) ).real, linestyle='dotted',)
# In[]
def _error(xr, xi, rr, ri, 
           xlim):
    # xv = np.linspace(xlim[0], xlim[1], 100000)
    _zi[:] = xr 
    _zi[:]+= 1j * xi
    _R[:]  = rr 
    _R[:] += 1j * ri
    #out = FD_expanded_v2(_xv, _zi, 1.0, 0.0, _R)
    #_fi[:] = out
    FD_expanded_v2_opt(_xv, _zi, 1.0, 0.0, _R, _fi)
    # _fi = FD_expanded_v2(_xv, _zi, 1.0, 0.0, _R)
    diff = _fxv - _fi
    deriv = np.diff(_fi) / _dxv
    t1 = np.trapezoid(np.abs(diff), x=_xv)
    t2 = np.trapezoid(np.abs(_fid - deriv), x = _xv_der)
    #t2 *= 2.0
    t1 += t2
    return t1
# rgain = r.gain()
class track:
    def __init__(self):
        self.it = 0
    def count(self):
        self.it += 1

Track = track()

def error(p):
    n = len(p)//4
    res = _error(p[0:n], 
                 p[n:2*n], 
                 p[2*n:3*n],
                 p[3*n:4*n],
                 xlim)
    if np.mod(Track.it, 100) == 0:
        print(res)
    Track.count()
    return res

p = np.hstack((xph.real, 
               xph.imag, 
               Rph.real, 
               Rph.imag
               )) + np.random.random(4*len(xph))*0.0001

n = npoles
# _xv = np.linspace(xlim[0], xlim[1], 100000)
npoints = 500
_xv1 = np.linspace(xlim[0], -10,npoints)
_xv2 = np.linspace(-10 + 20/npoints, 10- 20/npoints,     npoints)
_xv3 = np.linspace(10, xlim[1], npoints)
_xv  = np.hstack((_xv1, _xv2, _xv3))
_dxv = np.diff(_xv)
_xv_der = (_xv[1:] + _xv[:-1])/2 

dx   = 1e-3
_fid = (f(_xv_der + dx/2) - f(_xv_der - dx/2) ) / dx 

_con_x = np.linspace(2*xlim[0], 2*xlim[1], npoints)

def Maxfunc(p):
    _zi[:] = p[0:n]
    _zi[:]+= 1j * p[n:2*n]
    _R[:]  = p[2*n:3*n]
    _R[:] += 1j * p[3*n:4*n]
    _F     = FD_expanded_v2(_con_x, _zi, 1.0, 0.0, _R).real
    return 1.0 - _F.max()
def Minfunc(p):
    _zi[:] = p[0:n]
    _zi[:]+= 1j * p[n:2*n]
    _R[:]  = p[2*n:3*n]
    _R[:] += 1j * p[3*n:4*n]
    _F     = FD_expanded_v2(_con_x, _zi, 1.0, 0.0, _R).real
    return _F.min()

_zi = np.zeros(n, dtype=complex)
_R  = np.zeros(n, dtype=complex)
_fi = np.zeros(len(_xv),dtype=complex)
_fxv= f(_xv)
# def callback(p,k, accept):
#     _zi[:] = p[0:n]
#     _zi[:]+= 1j * p[n:2*n]
#     _R[:]  = p[2*n:3*n]
#     _R[:] += 1j * p[3*n:4*n]
#     plt.plot(_xv*1.25,  f(_xv*1.25)- FD_expanded_v2(_xv*1.25, Xiout, 1.0, 0.0, Riout) + 0.5)
#     plt.show()

print(error(p))
# have_imag = True
bounds =[(-500.0, 500.0)] * n  + [(0.1,3000)] * n \
         + [(-8000,8000)] * n  + [(-0, 0.0)] * n
from scipy.optimize import minimize, basinhopping

# In[]
cons = [{'type':'ineq', 'fun':Maxfunc},
        {'type':'ineq', 'fun':Minfunc}, ]

# if np.mod(n,2)==0:
#     for i in range(npoles // 2):
#         cons += [{'type': 'eq', 'fun':lambda x: x[2*i:2*i+2].sum()}]
# if np.mod(n,2)==1:
#     for i in range(npoles // 2):
#         cons += [{'type': 'eq', 'fun':lambda x: x[2*i:2*i+2].sum()}]
#     cons += [{'type': 'eq', 'fun':lambda x: x[i+2]}]


res = minimize(error, p, method='trust-constr', 
               bounds      = bounds,
               constraints = cons,
               tol         = 1e-6,
               options     = {# 'gtol': 1e-6, 
                              'disp': True, 
                              'maxiter':500,
                              },)
Xiout  = res.x[0:n]     + 1j * res.x[n:2*n  ]
Riout  = res.x[2*n:3*n] + 1j * res.x[3*n:4*n]
error(res.x)
# deriv = np.diff(_fi) / _dxv

# res = basinhopping(error, p, niter = 50,callback =callback, 
#                    minimizer_kwargs={'method':'SLSQP', 
#                                      'bounds':bounds,
#                                      'constraints': cons, 
#                                      'tol':1e-6,
#                                      'options':{'gtol': 1e-6, 
#                                                 'disp': True, 
#                                                 'maxiter': 1000,
#                                                 }
#                                      })

# In[]
plt.show()
plt.semilogy(xv, np.abs(f(xv) - FD_expanded_v2(xv, Xiout, 1.0, 0.0, Riout)+0.5)); 
plt.semilogy(xv, np.abs(f(xv) - FD_expanded_v2(xv, xph, 1.0, 0.0, Rph)+0.5));
plt.title(r'Logplot of $|f^<(x) - f_{approx}(x)|$')
