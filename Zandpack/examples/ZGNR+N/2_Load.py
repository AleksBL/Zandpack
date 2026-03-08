#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:01:23 2023

@author: aleksander
"""

from pickle import load
from Zandpack.TimedependentTransport import TD_Transport as TDT
import numpy as np
import matplotlib.pyplot as plt

Dev = load(open('Dev.SiP','rb'))

gd   = Dev.to_sisl()
ge1  = Dev.elecs[0].to_sisl()
ge2  = Dev.elecs[1].to_sisl()

R    = TDT([ge1,ge2], gd,
           kT_i=[.025, .025], 
           mu_i = [.0, .0])
line = np.linspace(-7,7,151) + 1j*1e-2 + 1e-3
line = np.vstack((line,line))
R.Make_Contour(line, 4)
Dev.custom_tbtrans_contour = R.Contour
Dev.fdf()
Dev.run_tbt_analyze_in_dir()
Dev.run_tbtrans_in_dir(overwrite_prev=True)
# In[]
cC,cH,cAl = 4,1,4
sub_idx   = []
it        = 0

for i in range(len(Dev.s)):
    if Dev.s[i]==6 or Dev.s[i]==7 :
        sub_idx += [it+2]
        it += cC
    elif Dev.s[i]==1:
        it += cH
    #elif Dev.s[i]==13:
    #    sub_idx += [it,it+1,it+2,it+3]
    #    it += cAl

R.Device = Dev
R.read_data(sub_orbital=sub_idx)
R.pickle('ZGNR')
