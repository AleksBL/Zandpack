#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sawtooth version of Bias_NO.py: same dH/bias as Bias_saw.py, transformed into
the nonorthogonal basis.  nozand imports this file under the module name
"Bias", so it must not import from Bias.py / Bias_saw.py themselves.
"""

from Zandpack.Help import TDHelper
from shared import dH_saw as dH_orthogonal, delta_saw as delta

Hlp = TDHelper('synth')
iL  = Hlp.invLowdin

def dH(t, sigma):
    return iL @ dH_orthogonal(t, sigma) @ iL

bias = delta

def dissipator(t, s):
    return 0.0
