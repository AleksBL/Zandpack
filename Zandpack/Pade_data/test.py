#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 13:12:09 2026

@author: aleks
"""

import numpy as np
import baryrat
from Zandpack.PadeDecomp import FD_expanded, Hu_poles
from Zandpack.plot import plt

xlims = (-200,200)
kT = 1.0
def f(x): return 1/(1 + np.exp(x)) - 1/2

npoles = 12
r = baryrat.brasil(f, xlims, 2 * npoles,2000)
x = np.linspace(xlims[0] - 200,xlims[1] + 200, 1000)
xp, Rp = r.polres()
idx    = np.where(xp.imag>1e-10)[0]
xp     = xp[idx]
Rp     = Rp[idx]
plt.plot(x,f(x), label = 'Fermi func');
# plt.plot(x,r(x), label = 'brasil r', linestyle='dashed')
plt.plot(x, FD_expanded(x, xp, 1/kT, coeffs = -Rp) - 1/2, label = 'brasil FDE' , linestyle='dashed')
xph, Rph = Hu_poles(npoles)
plt.plot(x, FD_expanded(x, xph, 1/kT, coeffs = Rph) - 1/2, label = 'JieHu2011')
plt.legend()


