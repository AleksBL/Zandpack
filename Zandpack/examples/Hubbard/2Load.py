#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:47:25 2023

@author: aleksander
"""

from Zandpack.TimedependentTransport import TD_Transport as TDT
import numpy as np
from pickle import load

D = load(open("Dev.SiP", "rb"))
ge1, ge2 = D.elecs[0].to_sisl(), D.elecs[1].to_sisl()
gd = D.to_sisl()
#D.elecs[0].sl = 'Hubbard_' + D.elecs[0].sl
#D.elecs[1].sl = 'Hubbard_' + D.elecs[1].sl

R  = TDT([ge1, ge2], gd, kT_i=[0.025, 0.025])
line = np.linspace(-8, 8, 200) + 2e-2j + 1e-3
line = np.vstack((line, line))
R.Make_Contour(line, 15)
D.calculate_hubbard_transport(R.Contour)

R.Device = D
R.read_data(D_lead_ortho=False) # Using the manual calculator gives 
                                # a format where we need to not attempt to 
                                # calculate the overlap corrections, 
                                # even though they are zero
R.pickle("Hub")
