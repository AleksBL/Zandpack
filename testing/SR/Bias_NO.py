#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias and dH for the nonorthogonal (nozand) run.

dH is transformed from the orthogonal (Lowdin) basis into the nonorthogonal
one:  H_NO = iL @ H_O @ iL,  iL = invLowdin.  The bias is basis independent.

Note: nozand imports this file under the module name "Bias", so it must not
import from Bias.py itself (that would be a circular self-import).
"""

from Zandpack.Help import TDHelper
from shared import dH as dH_orthogonal, delta

Hlp = TDHelper('STM2')
iL  = Hlp.invLowdin

def dH(t, sigma):
    return iL @ dH_orthogonal(t, sigma) @ iL

bias = delta

def dissipator(t, s):
    return 0.0
