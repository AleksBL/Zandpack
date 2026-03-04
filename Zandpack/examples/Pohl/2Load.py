#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:15:08 2023

@author: aleksander
"""

from Zandpack.TimedependentTransport import TD_Transport as TDT
import numpy as np
from pickle import load

D = load(open("Dev.SiP", "rb"))
ge1, ge2 = D.elecs[0].to_sisl(), D.elecs[1].to_sisl()
gd = D.to_sisl()

R    = TDT([ge1, ge2], gd, kT_i=[0.05, 0.05])
line = np.linspace(-6, 6, 151) + 3e-2j + 1e-3
line = np.vstack((line, line))
R.Make_Contour(line, 12)
D.custom_tbtrans_contour = R.Contour
D.fdf()
D.run_tbtrans_in_dir()
R.Device = D
R.read_data()
R.pickle("Pohl")