#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:56:07 2023

@author: aleksander
"""

from pickle import load
from Zandpack.TimedependentTransport import TD_Transport as TDT
import numpy as np
import matplotlib.pyplot as plt
from Zandpack.FittingTools import piecewise_linspace

Dev  = load(open('Dev.SiP','rb'))
gd   = Dev.to_sisl()
ge1  = Dev.elecs[0].to_sisl()
ge2  = Dev.elecs[1].to_sisl()
R    = TDT([ge1,ge2], gd,kT_i=[.025, .025], mu_i = [.0, .0])
line = np.linspace(-8,8,101) + 1j*1e-2 + 1e-3
line = np.vstack((line,line))
R.Make_Contour(line,2)
Dev.custom_tbtrans_contour = R.Contour
Dev.fdf()
Dev.run_tbt_analyze_in_dir()
Dev.run_tbtrans_in_dir(DOS_GF=True, overwrite_prev=True)

sub_idx = []
cC,cAu  = 4,6
it      = 0
R.Device = Dev
R.read_data()
R.pickle('C60', compression='gzip')
