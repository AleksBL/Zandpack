#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias and dH for the nonorthogonal (nozand) run.

`nozand` propagates in the nonorthogonal basis, so here **both** sides of `dH`
have to be transformed -- this is what makes this test different from the ones
whose `dH` ignores `sigma`:

    argument:   sigma_NO -> sigma_O = iL @ sigma_NO @ iL   (iL = invLowdin)
    result:     H_O      -> H_NO    = iL @ H_O @ iL

With a sigma-independent `dH` the argument transform is a no-op, which is why
the other `Bias_NO.py` files only transform the result.

Note: nozand imports this file under the module name "Bias", so it must not
import from Bias.py itself (circular self-import).
"""

from Zandpack.Help import TDHelper
from shared import dH as dH_orthogonal, delta

Hlp = TDHelper('synth_dm')
iL  = Hlp.invLowdin

def dH(t, sigma_NO):
    sigma_O = iL @ sigma_NO @ iL
    return iL @ dH_orthogonal(t, sigma_O) @ iL

bias = delta

def dissipator(t, s):
    return 0.0
