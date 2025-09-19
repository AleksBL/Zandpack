#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:23:16 2022

@author: aleksander
"""
import numpy as np
pi = np.pi
                 # gamma,   beta1, beta2,   A,        B,      C,      D
unpol = np.array([-0.1423, 1.0529, 0.3334,  0.0311,  -0.048,  0.002, -0.0116])
pol   = np.array([-0.0843, 1.3981, 0.2611,  0.01555, -0.0269, 0.0007,-0.0048])
diff  = pol - unpol

def f(pol):
    return ((1 + pol)**(4/3) + (1-pol)**(4/3)-2)/(2**(4/3) - 2)

def _Vxc_PZ(n1, n2 = None):
    # See Appendix 4 in http://homes.nano.aau.dk/tgp/master_2022.pdf
    # n1,n2 (n,) numpy arrays of density
    ntot = n1 if n2 is None else n1+n2
    pol  = np.zeros(ntot.shape) if n2 is None else (n1 - n2)/ntot
    rs = (3/(4*np.pi * ntot))**(1/3)
    Vxc = (-(9/(4*pi**2))**(1/3))/rs
    
    idx1 = rs >  1
    idx2 = rs <= 1
    
    vals = unpol[:,None]  + f(pol)[None,:] * diff[:,None]
    gamma, b1, b2, A, B, C, D = vals
    
    #print(idx1, idx2, vals)
    T1  = gamma[idx1]*(1 + 7/6 * b1[idx1]*np.sqrt(rs[idx1]) + 4/3 * b2[idx1] * rs[idx1])
    T1 /= (1 + b1[idx1]*np.sqrt(rs[idx1]) + b2[idx1] * rs[idx1])**2
    Vxc[idx1] += T1
    
    T2 = A[idx2]*np.log(rs[idx2]) + B[idx2] -1/3 * A[idx2]
    T2+= 2/3*C[idx2]*rs[idx2]*np.log(rs[idx2])
    T2+=(2/3*D[idx2] - 1/3*C[idx2] ) * rs[idx2]
    Vxc[idx2] += T2

    return Vxc

def Vxc_PZ(n1, n2 = None):
    ntot = n1 if n2 is None else n1+n2
    idx = np.where(ntot>0)
    if n2 is None:
        res = _Vxc_PZ(n1[idx], n2)
    else:
        res = _Vxc_PZ(n1[idx], n2[idx])
    out = np.zeros(ntot.shape)
    out[idx] = res
    return out

        
    
