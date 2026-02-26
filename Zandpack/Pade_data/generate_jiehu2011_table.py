#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:03:16 2026

@author: aleks
"""
import numpy as np
from Zandpack.PadeDecomp import FD_expanded, Hu_poles

kT = 1.0
def f(x): return 1/(1 + np.exp(x)) - 1/2
def find_dev(f2, tol = 1e-5):
    dx = 0.025
    dev = 0.1* tol
    x = 0.05 - dx
    it = 0
    while dev < tol:
        if it < 10:
            x += dx
        else:
            x += 0.01 * x
        dev = max((np.abs(f(x ) - f2(x )), 
                   np.abs(f(-x) - f2(-x))
                  )
                 )
        it += 1
    if it <= 10:
        x -= dx
    else:
        x -= 0.01*x
    return x
pots  = []
print('Constructing Table')
allpots = range(2,14)
allnl   = range(0,100)
table = np.zeros((len(allpots),100)) * np.nan
for i,pot in enumerate(allpots):
    nls   = []
    for j,nl in enumerate(allnl):
        xph, Rph = Hu_poles(nl)
        def ftest(x):
            return FD_expanded(np.array([x]), xph, 1/kT, coeffs = Rph)[0] - 0.5
        table[i,j] = find_dev(ftest, tol = 10**-pot)
        nls += [nl]
    pots += [ -pot ]

np.savez_compressed('JieHu2011_accuracy_table.npz',
                     table = table,
                     pots  = pots,
                     nl    = allnl)





