#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:25:48 2024

@author: aleksander
"""

from pickle import load
from Zandpack.TimedependentTransport import TD_Transport as TDT
import numpy as np
import matplotlib.pyplot as plt
from Zandpack.FittingTools import piecewise_linspace
# assert 1 ==0
Ng = 0
Dev  = load(open('Dev'+str(Ng)+'.SiP','rb'))
gd   = Dev.to_sisl()
ge1  = Dev.elecs[0].to_sisl()
ge2  = Dev.elecs[1].to_sisl()
R    = TDT([ge1,ge2], gd,kT_i=[.025, .025], mu_i = [.0, .0])
line = np.linspace(-8,8,251) + 1j*1.5e-2 + 1e-3
line = np.vstack((line,line))
R.Make_Contour(line,22)
Dev.custom_tbtrans_contour = R.Contour
Dev.fdf(eta = 1.5e-2)
Dev.run_tbt_analyze_in_dir()
Dev.run_tbtrans_in_dir(DOS_GF=True)
R.Device = Dev
R.read_data()
R.pickle('T2T_G'+str(Ng))