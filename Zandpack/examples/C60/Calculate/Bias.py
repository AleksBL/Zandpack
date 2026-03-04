#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:29:36 2022

@author: aleksander
"""

import numpy as np
import sisl
from Zandpack.Help import TDHelper
from Initial import name
# Template for the Bias.py file
# Needed for running the zand program

Hlp = TDHelper(name)

def dH(t,sigma, return_raw_H = False):
    # Lead-device overlap correction
    DynCor= Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in [0,1]])
    # Your density dependence here
    # .....
    return DynCor # + more

def bias(t,a):
    V = 0.5 * np.sin(0.1 * t) * np.exp(-((t - 0.0)**2/25**2))
    if a == 0:
        return -V
    else:
        return +V

def dissipator(t,sig):
    return 0.0
